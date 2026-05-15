#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/techniques/gi/Reservoir.glsl"

layout(local_size_x = 256) in;

layout(rgba32ui) uniform restrict uimage2D uimg_rgba32ui;

/*const*/
#if PASS_INDEX == 0
#define REUSETEX usam_restirReuseTex0
#elif PASS_INDEX == 1
#define REUSETEX usam_restirReuseTex1
#elif PASS_INDEX == 2
#define REUSETEX usam_restirReuseTex2
#elif PASS_INDEX == 3
#define REUSETEX usam_restirReuseTex3
#elif PASS_INDEX == 4
#define REUSETEX usam_restirReuseTex4
#elif PASS_INDEX == 5
#define REUSETEX usam_restirReuseTex5
#elif PASS_INDEX == 6
#define REUSETEX usam_restirReuseTex6
#else
#define REUSETEX usam_restirReuseTex7
#endif
/*const*/

bool restir_updateReservoirM(inout float reservoirM, inout float wSum, float wi, float m, float rand) {
    wSum += wi;
    reservoirM += m;
    return rand < wi / wSum;
}



ShiftMapping evaluateShiftMapping(
ivec2 texelDST,
ReSTIRReservoir canonResSRC,
SpatialSampleData sampleDST, SpatialSampleData sampleSRC,
vec3 viewPosDST, vec3 viewPosSRC
) {
    const float EPSILON = 1e-6;
    ShiftMapping mapping = shiftMapping_init();

    if (canonResSRC.Y.w > EPSILON && restir_isReservoirValid(canonResSRC)) {
        vec3 hitViewPosSRC = viewPosSRC + canonResSRC.Y.xyz * canonResSRC.Y.w;
        vec3 diffSRCtoDST = hitViewPosSRC - viewPosDST;
        float dist2 = dot(diffSRCtoDST, diffSRCtoDST);
        if (dist2 > EPSILON) {
            vec3 dirSRCtoDST = diffSRCtoDST * inversesqrt(dist2);
            float cosSRC = dot(sampleSRC.normal, canonResSRC.Y.xyz);
            float cosPhiSRC = -dot(canonResSRC.Y.xyz, sampleSRC.hitNormal);
            float cosPhiDST = -dot(dirSRCtoDST, sampleSRC.hitNormal);
            if (cosPhiSRC > 0.0 && cosPhiDST > 0.0) {
                vec3 VDST = normalize(-viewPosDST);
                vec4 resampleMaterialDataDST = transient_restir_resampleMaterial_fetch(texelDST);
                ResampleMaterial matDST = resampleMaterial_unpack(resampleMaterialDataDST);
                float pHat = evalTargetFunction(sampleSRC.sampleValue.xyz, sampleDST.normal, dirSRCtoDST, VDST, matDST);
                if (pHat > 0.0) {
                    float jacobian_DST = clamp(((canonResSRC.Y.w * canonResSRC.Y.w) * cosPhiDST) / (dist2 * cosPhiSRC), 0.0, 256.0);
                    mapping.Y = vec4(dirSRCtoDST, sqrt(dist2));
                    mapping.targetPHat = pHat * jacobian_DST;
                    if (cosSRC > 0.0) {
                        mapping.reusableTargetPHat = mapping.targetPHat;
                    }
                }
            }
        }
    }

    return mapping;
}

void doResample(
ivec2 texelDST, ivec2 texelSRC,
ReSTIRReservoir canonResDST, ReSTIRReservoir canonResSRC,
SpatialSampleData sampleDST, SpatialSampleData sampleSRC,
ShiftMapping srcToDst, ShiftMapping dstToSrc
) {
    if (shiftMapping_isReusable(srcToDst)) {
        uvec4 pairwiseMISMetadataDST = transient_restir_pairwiseMISMetadata_fetch(texelDST);
        PairwiseMISMetadata metaDST = pairwiseMISMetadata_unpack(pairwiseMISMetadataDST);
        float accumMDST = metaDST.accumM;

        float rcMDivK_DST = canonResDST.m / SETTING_GI_SPATIAL_REUSE_COUNT;
        float MiPiRiY = canonResSRC.m * sampleSRC.sampleValue.w;
        float mi_DST = MiPiRiY * safeRcp(MiPiRiY + rcMDivK_DST * srcToDst.reusableTargetPHat);

        float mcIncrement_DST = 1.0;
        if (shiftMapping_hasTarget(dstToSrc)) {
            float MiPiRcY = canonResSRC.m * dstToSrc.targetPHat;
            mcIncrement_DST = 1.0 - MiPiRcY * safeRcp(MiPiRcY + rcMDivK_DST * sampleDST.sampleValue.w);
        }

        metaDST.mc += mcIncrement_DST;
        metaDST.numValidNeighbors += 1u;

        float neighborWi = srcToDst.reusableTargetPHat * max(canonResSRC.avgWY, 0.0) * mi_DST;
        float spatialWSumDST = metaDST.spatialWSum;
        float neighborRand = rand_stbnVec1(rand_newStbnPos(texelDST, RANDOM_FRAME / 64u + 4u + PASS_INDEX), RANDOM_FRAME);
        if (restir_updateReservoirM(accumMDST, spatialWSumDST, neighborWi, canonResSRC.m, neighborRand)) {
            metaDST.selectedTexel = texelSRC;
        }
        metaDST.accumM = accumMDST;
        metaDST.spatialWSum = spatialWSumDST;
        transient_restir_pairwiseMISMetadata_store(texelDST, pairwiseMISMetadata_pack(metaDST));
    }
}

void main() {
    ivec2 localFetchPos = ivec2(gl_GlobalInvocationID.xy) & RESTIR_REUSE_TEX_MASK;
    ivec2 tileId = ivec2(gl_GlobalInvocationID.xy) >> RESTIR_REUSE_TEX_BITS;
    ivec2 tileOrigin = tileId * RESTIR_REUSE_TILE_SIZE;
    uvec4 pairData = texelFetch(REUSETEX, localFetchPos, 0);
    ivec2 localA = ivec2(pairData.xy);
    ivec2 localB = ivec2(pairData.zw);
    ivec2 localD = localB - localA;
    localD = ((localD + RESTIR_REUSE_TILE_SIZE_HALF) & RESTIR_REUSE_TILE_MASK) - RESTIR_REUSE_TILE_SIZE_HALF;
    localB = localA + localD;
    localA = (localA + uval_restirSpatialTileOffset);
    localB = (localB + uval_restirSpatialTileOffset);
    ivec2 texelA = tileOrigin + localA;
    ivec2 texelB = tileOrigin + localB;
    uint validA = uint(all(lessThan(ivec4(texelA, ivec2(-1)), ivec4(uval_mainImageSizeI, texelA))));
    uint validB = uint(all(lessThan(ivec4(texelB, ivec2(-1)), ivec4(uval_mainImageSizeI, texelB))));

    if (bool(validA & validB & uint(texelA != texelB))){
        float viewZA = texelFetch(usam_gbufferSolidViewZ, texelA, 0).x;
        float viewZB = texelFetch(usam_gbufferSolidViewZ, texelB, 0).x;
        if (viewZA > -65536.0 && viewZB > -65536.0) {
            uvec4 spatialSamplePackedDataA = transient_restir_spatialInput_fetch(texelA);
            uvec4 spatialSamplePackedDataB = transient_restir_spatialInput_fetch(texelB);

            SpatialSampleData sampleA = spatialSampleData_unpack(spatialSamplePackedDataA);
            SpatialSampleData sampleB = spatialSampleData_unpack(spatialSamplePackedDataB);

            if (dot(sampleA.geomNormal, sampleB.geomNormal) > 0.99) {
                uvec4 repA;
                uvec4 repB;
                if (bool(frameCounter & 1)) {
                    repA = history_restir_reservoirTemporal1_fetch(texelA);
                    repB = history_restir_reservoirTemporal1_fetch(texelB);
                } else {
                    repA = history_restir_reservoirTemporal2_fetch(texelA);
                    repB = history_restir_reservoirTemporal2_fetch(texelB);
                }

                vec2 screenPosA = coords_texelToUV(texelA, uval_mainImageSizeRcp);
                vec3 viewPosA = coords_toViewCoord(screenPosA, viewZA, global_camProjInverse);
                vec2 screenPosB = coords_texelToUV(texelB, uval_mainImageSizeRcp);
                vec3 viewPosB = coords_toViewCoord(screenPosB, viewZB, global_camProjInverse);

                ReSTIRReservoir canonResA = restir_reservoir_unpack(repA);
                ReSTIRReservoir canonResB = restir_reservoir_unpack(repB);

                ShiftMapping shiftBtoA = evaluateShiftMapping(texelA, canonResB, sampleA, sampleB, viewPosA, viewPosB);
                ShiftMapping shiftAtoB = evaluateShiftMapping(texelB, canonResA, sampleB, sampleA, viewPosB, viewPosA);

                doResample(texelA, texelB, canonResA, canonResB, sampleA, sampleB, shiftBtoA, shiftAtoB);
                doResample(texelB, texelA, canonResB, canonResA, sampleB, sampleA, shiftAtoB, shiftBtoA);
            }
        }
    }
}
