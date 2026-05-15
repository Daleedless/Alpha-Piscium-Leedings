#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "/util/Material.glsl"
#include "/util/ThreadGroupTiling.glsl"
#include "/techniques/SST2.glsl"
#include "/techniques/gi/Common.glsl"
#include "/techniques/gi/Reservoir.glsl"
#include "/techniques/HiZCheck.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(std430, binding = 5) buffer RayData {
    uvec4 ssbo_rayData[];
};

layout(std430, binding = 6) buffer RayIndexData {
    uint ssbo_rayDataIndices[];
};

layout(rgba16f) uniform image2D uimg_rgba16f;
layout(rgb10_a2) uniform restrict writeonly image2D uimg_rgb10_a2;
layout(r32f) uniform image2D uimg_r32f;
layout(rgba8) uniform restrict writeonly image2D uimg_temp5;

shared uint shared_rayCount[16];

void main() {
    sst_init(SETTING_GI_SST_THICKNESS);
    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    uvec2 binId = swizzledWGPos >> 1u;
    uint numBinX = (uval_mainImageSizeI.x + 31) >> 5;
    uint binIdx = binId.y * numBinX + binId.x;
    ivec2 binLocalPos = texelPos & 31;
    uint binLocalIndex = sst2_encodeBinLocalIndex(binLocalPos);
    uint binWriteBaseIndex = binIdx * 1024;
    uint dataIndex = binWriteBaseIndex + binLocalIndex;
    uint rayIndex = 0xFFFFFFFFu;

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        SpatialSampleData centerSampleData = spatialSampleData_unpack(transient_restir_spatialInput_fetch(texelPos));
        history_restir_prevSample_store(texelPos, centerSampleData.sampleValue);
        history_restir_prevHitNormal_store(texelPos, vec4(centerSampleData.hitNormal * 0.5 + 0.5, 0.0));
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos, 4, texelPos);

        if (viewZ > -65536.0) {
            vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
            vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
            vec3 V = normalize(-viewPos);

            uvec4 reprojectedData = bool(frameCounter & 1) ? history_restir_reservoirTemporal1_fetch(texelPos) : history_restir_reservoirTemporal2_fetch(texelPos);
            ReSTIRReservoir temporalReservoir = restir_reservoir_unpack(reprojectedData);

            ReSTIRReservoir spatialReservoir = restir_reservoir_unpack(transient_restir_spatialReservoirAccum_fetch(texelPos));
            uvec4 meta = transient_restir_pairwiseMISMetadata_fetch(texelPos);

            ivec2 winTexel = ivec2(unpackUInt2x16(meta.x));
            uint numValidNeighbors = meta.y;
            float mc = uintBitsToFloat(meta.z);
            float spatialWSum = uintBitsToFloat(meta.w);

            float rcAvgWY = max(temporalReservoir.avgWY, 0.0);
            float canonicalWi = centerSampleData.sampleValue.w * rcAvgWY * mc;
            float canonicalRand = rand_stbnVec1(rand_newStbnPos(texelPos, RANDOM_FRAME / 64u + 4u + 8u), RANDOM_FRAME);

            bool chooseCanon = restir_updateReservoir(
                spatialReservoir,
                spatialWSum,
                temporalReservoir.Y,
                canonicalWi,
                0.0,
                canonicalRand
            );

            vec4 selectedSampleF;
            if (chooseCanon || winTexel == texelPos) {
                selectedSampleF = centerSampleData.sampleValue;
            } else {
                SpatialSampleData winSample = spatialSampleData_unpack(transient_restir_spatialInput_fetch(winTexel));
                float winViewZ = texelFetch(usam_gbufferSolidViewZ, winTexel, 0).x;
                vec2 winScreenPos = coords_texelToUV(winTexel, uval_mainImageSizeRcp);
                vec3 winViewPos = coords_toViewCoord(winScreenPos, winViewZ, global_camProjInverse);

                uvec4 winRep = bool(frameCounter & 1) ? history_restir_reservoirTemporal1_fetch(winTexel) : history_restir_reservoirTemporal2_fetch(winTexel);
                ReSTIRReservoir winRes = restir_reservoir_unpack(winRep);

                vec3 winHitViewPos = winViewPos + winRes.Y.xyz * winRes.Y.w;
                vec3 diff = winHitViewPos - viewPos;
                float dist2 = dot(diff, diff);
                vec3 dir = diff * inversesqrt(dist2);

                GBufferData gData = gbufferData_init();
                gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, texelPos, 0), gData);
                gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelPos, 0), gData);
                Material material = material_decode(gData);

                float pHat = evalTargetFunction(winSample.sampleValue.xyz, centerSampleData.normal, dir, V, material);
                float cosPhiWin = -dot(winRes.Y.xyz, winSample.hitNormal);
                float cosPhiCenter = -dot(dir, winSample.hitNormal);

                float jacobian = clamp(((winRes.Y.w * winRes.Y.w) * cosPhiCenter) / (dist2 * cosPhiWin), 0.0, 256.0);
                selectedSampleF = vec4(winSample.sampleValue.xyz, pHat * jacobian);
            }

            vec4 ssgiDiffOut = vec4(0.0, 0.0, 0.0, -1.0);
            vec4 ssgiSpecOut = vec4(0.0, 0.0, 0.0, -1.0);
            ReSTIRReservoir resultReservoir = spatialReservoir;

            float avgWY = spatialWSum * safeRcp(selectedSampleF.w) * safeRcp(float(numValidNeighbors + 1u));
            resultReservoir.avgWY = avgWY;

            vec3 winL_out = resultReservoir.Y.xyz;
            float winHitDist = resultReservoir.Y.w;
            vec3 H_out = normalize(winL_out + V);

            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelPos, 0), gData);
            Material material = material_decode(gData);

            float outNDotL = saturate(dot(gData.normal, winL_out));
            float outNDotH = saturate(dot(gData.normal, H_out));
            float outLDotH = saturate(dot(winL_out, H_out));

            vec3 outFresnel = fresnel_evalMaterial(material, outLDotH);
            float lambertianBRDF = outNDotL * RCP_PI;
            float NDotV = saturate(dot(centerSampleData.normal, V));
            float ggxBRDF = bsdf_ggx(material, outNDotL, NDotV, outNDotH);

            vec3 diffuseWeight = material.dielectric * (1.0 - outFresnel) * lambertianBRDF;
            vec3 specularWeight = outFresnel * ggxBRDF;
            vec3 fullBRDF = diffuseWeight + specularWeight;
            vec3 diffRatio3 = diffuseWeight * safeRcp(fullBRDF);

            vec3 totalOutput = selectedSampleF.xyz * fullBRDF * avgWY;
            ssgiDiffOut = vec4(totalOutput * diffRatio3, winHitDist);
            ssgiSpecOut = vec4(totalOutput * (vec3(1.0) - diffRatio3), winHitDist);
            vec3 specBrdf = texture(usam_specBRDFLUT, vec2(NDotV, material.roughness)).rgb;
            vec3 specAlbedo = saturate(material.f0RGB * specBrdf.x + material.f82TintRGB * specBrdf.y + specBrdf.z);
            ssgiSpecOut.rgb *= safeRcp(specAlbedo);

            #if SETTING_DEBUG_OUTPUT
            vec4 vvv = vec4(0.0);
            #endif
            if (!chooseCanon && winTexel != texelPos) {
                #if SETTING_DEBUG_OUTPUT
                vvv = vec4(0.0, 1.0, 0.0, 0.0);
                #endif

                SSTRay sstRay;
                if (spatialReservoir.Y.w > 0.0) {
                    vec3 expectHitViewPos = viewPos + spatialReservoir.Y.xyz * spatialReservoir.Y.w;
                    vec3 rayOrigin = coords_viewToScreen(viewPos, global_camProj);
                    vec3 rayEnd = coords_viewToScreen(expectHitViewPos, global_camProj);
                    vec4 rayDirLen = normalizeAndLength(rayEnd - rayOrigin);
                    sstRay = sstray_setup(texelPos, rayOrigin, rayDirLen.xyz, rayDirLen.w);
                } else {
                    sstRay = sstray_setup(texelPos, viewPos, spatialReservoir.Y.xyz);
                }
                sst_trace(sstRay, 4);
                if (sstRay.currT > 0.0) {
                    uvec4 packedData = sstray_pack(sstRay);
                    ssbo_rayData[dataIndex] = packedData;
                    rayIndex = sst2_encodeRayIndexBits(binLocalIndex, sstRay);
                } else {
                    bool discardSptialReuse = true;
                    if (sstRay.currT < -0.99) discardSptialReuse = false;

                    if (discardSptialReuse) {
                        resultReservoir = restir_initReservoir();
                        ssgiDiffOut = vec4(0.0);
                        ssgiSpecOut = vec4(0.0);
                        #if SETTING_DEBUG_OUTPUT
                        imageStore(uimg_temp5, texelPos, vec4(0.0, 0.0, 1.0, 0.0));
                        #endif
                    }
                }
            }
            #if SETTING_DEBUG_OUTPUT
            imageStore(uimg_temp5, texelPos, vvv);
            #endif

            ssgiDiffOut.rgb = clamp(ssgiDiffOut.rgb, 0.0, FP16_MAX);
            ssgiSpecOut.rgb = clamp(ssgiSpecOut.rgb, 0.0, FP16_MAX);
            transient_ssgiDiffOut_store(texelPos, ssgiDiffOut);
            transient_ssgiSpecOut_store(texelPos, ssgiSpecOut);
        }
    }
    ssbo_rayDataIndices[dataIndex] = rayIndex;
    uvec4 subgroupRayCountBalllot = subgroupBallot(rayIndex < 0xFFFFFFFFu);
    if (subgroupElect()) {
        shared_rayCount[gl_SubgroupID] = subgroupBallotBitCount(subgroupRayCountBalllot);
    }
    barrier();
    if (gl_SubgroupID == 0u) {
        uint partialRayCount = gl_SubgroupInvocationID < gl_NumSubgroups ? shared_rayCount[gl_SubgroupInvocationID] : 0u;
        uint totalRayCount = subgroupAdd(partialRayCount);
        if (subgroupElect()) {
            transient_spatialReuseRayCount_store(ivec2(swizzledWGPos), vec4(float(totalRayCount)));
        }
    }
}
