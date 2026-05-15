#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "/util/GBufferData.glsl"
#include "/util/Material.glsl"
#include "/util/Rand.glsl"
#include "/util/Mat2.glsl"
#include "/techniques/gi/Reservoir.glsl"

layout(local_size_x = 256) in;

layout(r32f) uniform restrict image2D uimg_r32f;
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

void doResample(
ivec2 texelDST, ivec2 texelSRC,
ReSTIRReservoir canonResDST, ReSTIRReservoir canonResSRC,
SpatialSampleData sampleDST, SpatialSampleData sampleSRC,
ShiftMapping srcToDst, ShiftMapping dstToSrc
) {
    if (shiftMapping_isReusable(srcToDst)) {
        float accumMDST = transient_restir_spatialReservoirAccum_fetch(texelDST).x;
        uvec4 pairwiseMISMetadataDST = transient_restir_pairwiseMISMetadata_fetch(texelDST);

        float rcMDivK_DST = canonResDST.m / 8.0;
        float MiPiRiY = canonResSRC.m * sampleSRC.sampleValue.w;
        float mi_DST = MiPiRiY * safeRcp(MiPiRiY + rcMDivK_DST * srcToDst.reusableTargetPHat);

        float mcIncrement_DST = 1.0;
        if (shiftMapping_hasTarget(dstToSrc)) {
            float MiPiRcY = canonResSRC.m * dstToSrc.targetPHat;
            mcIncrement_DST = 1.0 - MiPiRcY * safeRcp(MiPiRcY + rcMDivK_DST * sampleDST.sampleValue.w);
        }

        PairwiseMISMetadata metaDST = pairwiseMISMetadata_unpack(pairwiseMISMetadataDST);
        metaDST.mc += mcIncrement_DST;
        metaDST.numValidNeighbors += 1u;

        float neighborWi = srcToDst.reusableTargetPHat * max(canonResSRC.avgWY, 0.0) * mi_DST;
        float spatialWSumDST = metaDST.spatialWSum;
        float neighborRand = rand_stbnVec1(rand_newStbnPos(texelDST, RANDOM_FRAME / 64u + 4u + PASS_INDEX), RANDOM_FRAME);
        if (restir_updateReservoirM(accumMDST, spatialWSumDST, neighborWi, canonResSRC.m, neighborRand)) {
            metaDST.selectedTexel = texelSRC;
        }
        metaDST.spatialWSum = spatialWSumDST;
        transient_restir_spatialReservoirAccum_store(texelDST, vec4(accumMDST));
        transient_restir_pairwiseMISMetadata_store(texelDST, pairwiseMISMetadata_pack(metaDST));
    }
}

void main() {
    ivec2 localFetchPos = ivec2(gl_GlobalInvocationID.xy) % ivec2(256, 128);
    ivec2 tileId = ivec2(gl_GlobalInvocationID.xy) / ivec2(256, 128);
    ivec2 tileOrigin = tileId * ivec2(256, 256);
    uvec4 pairData = texelFetch(REUSETEX, localFetchPos, 0);
    ivec2 localA = ivec2(pairData.xy);
    ivec2 localB = ivec2(pairData.zw);
    ivec2 localD = localB - localA;
    localD = ((localD + 128) & 255) - 128;
    localB = localA + localD;
    localA = (localA + uval_restirSpatialTileOffset);
    localB = (localB + uval_restirSpatialTileOffset);
    ivec2 texelA = tileOrigin + localA;
    ivec2 texelB = tileOrigin + localB;
    bool validA = all(lessThan(ivec4(texelA, -1, -1), ivec4(uval_mainImageSizeI, texelA)));
    bool validB = all(lessThan(ivec4(texelB, -1, -1), ivec4(uval_mainImageSizeI, texelB)));

    if (validA && validB && texelA != texelB){
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

                ResampleMaterial matA = resampleMaterial_unpack(transient_restir_resampleMaterial_fetch(texelA));
                ShiftMapping shiftBtoA = evaluateShiftMapping(canonResB, matA, sampleA, sampleB, viewPosA, viewPosB);

                ResampleMaterial matB = resampleMaterial_unpack(transient_restir_resampleMaterial_fetch(texelB));
                ShiftMapping shiftAtoB = evaluateShiftMapping(canonResA, matB, sampleB, sampleA, viewPosB, viewPosA);

                doResample(texelA, texelB, canonResA, canonResB, sampleA, sampleB, shiftBtoA, shiftAtoB);
                doResample(texelB, texelA, canonResB, canonResA, sampleB, sampleA, shiftAtoB, shiftBtoA);
            }
        }
    }
}
