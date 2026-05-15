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

struct ShiftMapping {
    vec4 Y;
    float targetPHat;
    float reusableTargetPHat;
};

ShiftMapping shiftMapping_init() {
    ShiftMapping mapping;
    mapping.Y = vec4(0.0, 0.0, 0.0, -1.0);
    mapping.targetPHat = 0.0;
    mapping.reusableTargetPHat = 0.0;
    return mapping;
}

bool shiftMapping_hasTarget(ShiftMapping mapping) {
    return mapping.targetPHat > 0.0;
}

bool shiftMapping_isReusable(ShiftMapping mapping) {
    return mapping.reusableTargetPHat > 0.0;
}

ShiftMapping evaluateShiftMapping(
ReSTIRReservoir canonResSRC,
Material matDST,
SpatialSampleData sampleDST, SpatialSampleData sampleSRC,
vec3 viewPosDST, vec3 viewPosSRC
) {
    ShiftMapping mapping = shiftMapping_init();

    vec3 hitViewPosSRC = viewPosSRC + canonResSRC.Y.xyz * canonResSRC.Y.w;
    vec3 diffSRCtoDST = hitViewPosSRC - viewPosDST;
    float dist2 = dot(diffSRCtoDST, diffSRCtoDST);
    if (dist2 > 1e-6 && canonResSRC.Y.w > 1e-6 && restir_isReservoirValid(canonResSRC)) {
        vec3 dirSRCtoDST = diffSRCtoDST * inversesqrt(dist2);
        float cosSRC = dot(sampleSRC.normal, canonResSRC.Y.xyz);
        float cosPhiSRC = -dot(canonResSRC.Y.xyz, sampleSRC.hitNormal);
        float cosPhiDST = -dot(dirSRCtoDST, sampleSRC.hitNormal);
        if (cosPhiSRC > 0.0 && cosPhiDST > 0.0) {
            vec3 VDST = normalize(-viewPosDST);
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

    return mapping;
}

void applyShiftMapping(
ivec2 texelDST, ivec2 texelSRC,
inout ReSTIRReservoir accumResDST,
ReSTIRReservoir canonResDST, ReSTIRReservoir canonResSRC,
inout uvec4 metaDST,
SpatialSampleData sampleDST, SpatialSampleData sampleSRC,
ShiftMapping srcToDst, ShiftMapping dstToSrc
) {
    if (shiftMapping_isReusable(srcToDst)) {
        float rcMDivK_DST = canonResDST.m / 8.0;
        float MiPiRiY = canonResSRC.m * sampleSRC.sampleValue.w;
        float mi_DST = MiPiRiY * safeRcp(MiPiRiY + rcMDivK_DST * srcToDst.reusableTargetPHat);

        float mcIncrement_DST = 1.0;
        if (shiftMapping_hasTarget(dstToSrc)) {
            float MiPiRcY = canonResSRC.m * dstToSrc.targetPHat;
            mcIncrement_DST = 1.0 - MiPiRcY * safeRcp(MiPiRcY + rcMDivK_DST * sampleDST.sampleValue.w);
        }

        float mc_DST = uintBitsToFloat(metaDST.z) + mcIncrement_DST;
        metaDST.z = floatBitsToUint(mc_DST);
        metaDST.y += 1u;

        float neighborWi = srcToDst.reusableTargetPHat * max(canonResSRC.avgWY, 0.0) * mi_DST;
        float spatialWSumDST = uintBitsToFloat(metaDST.w);
        float neighborRand = rand_stbnVec1(rand_newStbnPos(texelDST, RANDOM_FRAME / 64u + 4u + PASS_INDEX), RANDOM_FRAME);
        if (restir_updateReservoir(accumResDST, spatialWSumDST, srcToDst.Y, neighborWi, canonResSRC.m, neighborRand)) {
            metaDST.x = (uint(texelSRC.y) << 16) | uint(texelSRC.x);
        }
        metaDST.w = floatBitsToUint(spatialWSumDST);
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

    ReSTIRReservoir accumResA = restir_initReservoir();
    ReSTIRReservoir accumResB = restir_initReservoir();

    uvec4 metaA = uvec4(0);
    uvec4 metaB = uvec4(0);
    #if PASS_INDEX == 0
    metaA = uvec4((uint(texelA.y) << 16) | uint(texelA.x), 0u, floatBitsToUint(1.0), 0u);
    metaB = uvec4((uint(texelB.y) << 16) | uint(texelB.x), 0u, floatBitsToUint(1.0), 0u);
    #else
    accumResA = restir_reservoir_unpack(transient_restir_spatialReservoirAccum_load(texelA));
    accumResB = restir_reservoir_unpack(transient_restir_spatialReservoirAccum_load(texelB));
    metaA = transient_restir_pairwiseMISMetadata_load(texelA);
    metaB = transient_restir_pairwiseMISMetadata_load(texelB);
    #endif

    if (validA && validB && texelA != texelB){
        float viewZA = texelFetch(usam_gbufferSolidViewZ, texelA, 0).x;
        float viewZB = texelFetch(usam_gbufferSolidViewZ, texelB, 0).x;
        if (viewZA > -65536.0 && viewZB > -65536.0) {
            uvec4 spatialSamplePackedDataA = transient_restir_spatialInput_fetch(texelA);
            uvec4 spatialSamplePackedDataB = transient_restir_spatialInput_fetch(texelB);
            uvec4 gbufferSolidData1A = texelFetch(usam_gbufferSolidData1, texelA, 0);
            uvec4 gbufferSolidData1B = texelFetch(usam_gbufferSolidData1, texelB, 0);
            uvec4 gbufferSolidData2A = texelFetch(usam_gbufferSolidData2, texelA, 0);
            uvec4 gbufferSolidData2B = texelFetch(usam_gbufferSolidData2, texelB, 0);
            uvec4 repA;
            uvec4 repB;
            if (bool(frameCounter & 1)) {
                repA = history_restir_reservoirTemporal1_fetch(texelA);
                repB = history_restir_reservoirTemporal1_fetch(texelB);
            } else {
                repA = history_restir_reservoirTemporal2_fetch(texelA);
                repB = history_restir_reservoirTemporal2_fetch(texelB);
            }

            SpatialSampleData sampleA = spatialSampleData_unpack(spatialSamplePackedDataA);
            SpatialSampleData sampleB = spatialSampleData_unpack(spatialSamplePackedDataB);

            GBufferData gDataA = gbufferData_init();
            gbufferData1_unpack(gbufferSolidData1A, gDataA);
            gbufferData2_unpack(gbufferSolidData2A, gDataA);
            Material matA = material_decode(gDataA);

            GBufferData gDataB = gbufferData_init();
            gbufferData1_unpack(gbufferSolidData1B, gDataB);
            gbufferData2_unpack(gbufferSolidData2B, gDataB);
            Material matB = material_decode(gDataB);

            ReSTIRReservoir canonResA = restir_reservoir_unpack(repA);
            ReSTIRReservoir canonResB = restir_reservoir_unpack(repB);

            #if PASS_INDEX == 0
            accumResA = canonResA;
            accumResB = canonResB;
            #endif

            if (dot(sampleA.geomNormal, sampleB.geomNormal) > 0.99) {
                vec2 screenPosA = coords_texelToUV(texelA, uval_mainImageSizeRcp);
                vec3 viewPosA = coords_toViewCoord(screenPosA, viewZA, global_camProjInverse);
                vec2 screenPosB = coords_texelToUV(texelB, uval_mainImageSizeRcp);
                vec3 viewPosB = coords_toViewCoord(screenPosB, viewZB, global_camProjInverse);

                ShiftMapping shiftAtoB = evaluateShiftMapping(canonResA, matB, sampleB, sampleA, viewPosB, viewPosA);
                ShiftMapping shiftBtoA = evaluateShiftMapping(canonResB, matA, sampleA, sampleB, viewPosA, viewPosB);
                applyShiftMapping(texelA, texelB, accumResA, canonResA, canonResB, metaA, sampleA, sampleB, shiftBtoA, shiftAtoB);
                applyShiftMapping(texelB, texelA, accumResB, canonResB, canonResA, metaB, sampleB, sampleA, shiftAtoB, shiftBtoA);
            }
        }
    }

    transient_restir_pairwiseMISMetadata_store(texelA, metaA);
    transient_restir_pairwiseMISMetadata_store(texelB, metaB);

    transient_restir_spatialReservoirAccum_store(texelA, restir_reservoir_pack(accumResA));
    transient_restir_spatialReservoirAccum_store(texelB, restir_reservoir_pack(accumResB));
}
