#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/techniques/gi/Reservoir.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32ui) uniform uimage2D uimg_rgba32ui;

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

void evaluateShift(
    ivec2 texelDST, ivec2 texelSRC,
    inout ReSTIRReservoir accumResDST,
    ReSTIRReservoir canonResDST, ReSTIRReservoir canonResSRC,
    inout uvec4 metaDST,
    Material matDST, Material matSRC,
    SpatialSampleData sampleDST, SpatialSampleData sampleSRC,
    vec3 viewPosDST, vec3 viewPosSRC
) {
    vec3 hitViewPosSRC = viewPosSRC + canonResSRC.Y.xyz * canonResSRC.Y.w;
    vec3 diffSRCtoDST = hitViewPosSRC - viewPosDST;
    float dist2 = dot(diffSRCtoDST, diffSRCtoDST);
    if (dist2 > 1e-6 && canonResSRC.Y.w > 1e-6 && restir_isReservoirValid(canonResSRC)) {
        vec3 dirSRCtoDST = diffSRCtoDST * inversesqrt(dist2);
        float cosSRC = dot(sampleSRC.normal, canonResSRC.Y.xyz);
        float cosPhiSRC = -dot(canonResSRC.Y.xyz, sampleSRC.hitNormal);
        float cosPhiDST = -dot(dirSRCtoDST, sampleSRC.hitNormal);
        if (cosSRC > 0.0 && cosPhiSRC > 0.0 && cosPhiDST > 0.0) {
            vec3 VDST = normalize(-viewPosDST);
            float pHat = evalTargetFunction(sampleSRC.sampleValue.xyz, sampleDST.normal, dirSRCtoDST, VDST, matDST);
            if (pHat > 0.0) {
                float jacobian_DST = clamp(((canonResSRC.Y.w * canonResSRC.Y.w) * cosPhiDST) / (dist2 * cosPhiSRC), 0.0, 256.0);
                float pcRiY_DST = pHat * jacobian_DST;
                float rcMDivK_DST = canonResDST.m / 8.0;
                float MiPiRiY = canonResSRC.m * sampleSRC.sampleValue.w;
                float mi_DST = MiPiRiY * safeRcp(MiPiRiY + rcMDivK_DST * pcRiY_DST);

                float mcIncrement_DST = 1.0;
                vec3 diffDSTtoSRC = (viewPosDST + canonResDST.Y.xyz * canonResDST.Y.w) - viewPosSRC;
                float dist2DSTtoSRC = dot(diffDSTtoSRC, diffDSTtoSRC);
                float cosPhiSRC_canon = -dot(canonResDST.Y.xyz, sampleDST.hitNormal);
                float RB2_canon = canonResDST.Y.w * canonResDST.Y.w;
                if (dist2DSTtoSRC > 1e-6 && RB2_canon >= 1e-6 && cosPhiSRC_canon > 0.0) {
                    vec3 cDirAtNbr = diffDSTtoSRC * inversesqrt(dist2DSTtoSRC);
                    float cCosPhiDST = -dot(cDirAtNbr, sampleDST.hitNormal);
                    if (cCosPhiDST > 0.0) {
                        float jacCn = clamp((RB2_canon * cCosPhiDST) / (dist2DSTtoSRC * cosPhiSRC_canon), 0.0, 256.0);
                        vec3 VSRC = normalize(-viewPosSRC);
                        float piRcY_SRC = evalTargetFunction(sampleDST.sampleValue.xyz, sampleSRC.normal, cDirAtNbr, VSRC, matSRC) * jacCn;
                        float MiPiRcY = canonResSRC.m * piRcY_SRC;
                        mcIncrement_DST = 1.0 - MiPiRcY * safeRcp(MiPiRcY + rcMDivK_DST * sampleDST.sampleValue.w);
                    }
                }

                float mc_DST = uintBitsToFloat(metaDST.z) + mcIncrement_DST;
                metaDST.z = floatBitsToUint(mc_DST);
                metaDST.y += 1u;
                float neighborWi = pcRiY_DST * max(canonResSRC.avgWY, 0.0) * mi_DST;
                float spatialWSumDST = uintBitsToFloat(metaDST.w);
                float neighborRand = rand_stbnVec1(rand_newStbnPos(texelDST, RANDOM_FRAME / 64u + 4u + PASS_INDEX), RANDOM_FRAME);
                vec4 newY = vec4(dirSRCtoDST, sqrt(dist2));
                if (restir_updateReservoir(accumResDST, spatialWSumDST, newY, neighborWi, canonResSRC.m, neighborRand)) {
                    metaDST.x = (uint(texelSRC.y) << 16) | uint(texelSRC.x);
                }
                metaDST.w = floatBitsToUint(spatialWSumDST);
            }
        }
    }
}

void main() {
    ivec2 localFetchPos = ivec2(gl_GlobalInvocationID.xy) % ivec2(256, 128);
    ivec2 tileId = ivec2(gl_GlobalInvocationID.xy) / ivec2(256, 128);
    ivec2 tileOrigin = tileId * ivec2(256, 256);
    uvec4 pairData = texelFetch(REUSETEX, localFetchPos, 0);
    ivec2 localA = ivec2(pairData.xy);
    ivec2 localB = ivec2(pairData.zw);
    localA = (localA + global_restirSpatialTileOffset) % 256;
    localB = (localB + global_restirSpatialTileOffset) % 256;
    ivec2 texelA = tileOrigin + localA;
    ivec2 texelB = tileOrigin + localB;
    bool validA = all(lessThan(texelA, uval_mainImageSizeI));
    bool validB = all(lessThan(texelB, uval_mainImageSizeI));
    if (!validA || !validB || texelA == texelB) return;

    GBufferData gDataA, gDataB;
    Material matA, matB;
    SpatialSampleData sampleA, sampleB;
    vec3 viewPosA, viewPosB;
    float viewZA, viewZB;
    ReSTIRReservoir canonResA, canonResB;
    ReSTIRReservoir accumResA = restir_initReservoir();
    ReSTIRReservoir accumResB = restir_initReservoir();
    uvec4 metaA = uvec4(0), metaB = uvec4(0);
    gDataA = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, texelA, 0), gDataA);
    gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelA, 0), gDataA);
    matA = material_decode(gDataA);
    sampleA = spatialSampleData_unpack(transient_restir_spatialInput_fetch(texelA));
    viewZA = texelFetch(usam_gbufferSolidViewZ, texelA, 0).x;
    if (viewZA <= -65536.0) return;
    vec2 screenPosA = coords_texelToUV(texelA, uval_mainImageSizeRcp);
    viewPosA = coords_toViewCoord(screenPosA, viewZA, global_camProjInverse);
    uvec4 repA = bool(frameCounter & 1) ? history_restir_reservoirTemporal1_fetch(texelA) : history_restir_reservoirTemporal2_fetch(texelA);
    canonResA = restir_reservoir_unpack(repA);
    #if PASS_INDEX == 0
    accumResA = canonResA;
    metaA = uvec4((uint(texelA.y) << 16) | uint(texelA.x), 0u, floatBitsToUint(1.0), 0u);
    #else
    accumResA = restir_reservoir_unpack(transient_restir_spatialReservoirAccum_load(texelA));
    metaA = transient_restir_pairwiseMISMetadata_load(texelA);
    #endif

    gDataB = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, texelB, 0), gDataB);
    gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelB, 0), gDataB);
    matB = material_decode(gDataB);
    sampleB = spatialSampleData_unpack(transient_restir_spatialInput_fetch(texelB));
    viewZB = texelFetch(usam_gbufferSolidViewZ, texelB, 0).x;
    if (viewZB <= -65536.0) return;
    vec2 screenPosB = coords_texelToUV(texelB, uval_mainImageSizeRcp);
    viewPosB = coords_toViewCoord(screenPosB, viewZB, global_camProjInverse);
    uvec4 repB = bool(frameCounter & 1) ? history_restir_reservoirTemporal1_fetch(texelB) : history_restir_reservoirTemporal2_fetch(texelB);
    canonResB = restir_reservoir_unpack(repB);
    #if PASS_INDEX == 0
    accumResB = canonResB;
    metaB = uvec4((uint(texelB.y) << 16) | uint(texelB.x), 0u, floatBitsToUint(1.0), 0u);
    #else
    accumResB = restir_reservoir_unpack(transient_restir_spatialReservoirAccum_load(texelB));
    metaB = transient_restir_pairwiseMISMetadata_load(texelB);
    #endif

    if (dot(sampleA.geomNormal, sampleB.geomNormal) > 0.99) {
        evaluateShift(texelA, texelB, accumResA, canonResA, canonResB, metaA, matA, matB, sampleA, sampleB, viewPosA, viewPosB);
        evaluateShift(texelB, texelA, accumResB, canonResB, canonResA, metaB, matB, matA, sampleB, sampleA, viewPosB, viewPosA);
    }

    transient_restir_spatialReservoirAccum_store(texelA, restir_reservoir_pack(accumResA));
    transient_restir_pairwiseMISMetadata_store(texelA, metaA);

    transient_restir_spatialReservoirAccum_store(texelB, restir_reservoir_pack(accumResB));
    transient_restir_pairwiseMISMetadata_store(texelB, metaB);
}
