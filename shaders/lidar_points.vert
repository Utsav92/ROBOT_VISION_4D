// ROBOT VISION 4D - LiDAR point renderer (GLSL MAT, vertex). Shared by lidar_points.frag (orange) and
// depth_error.frag (diagnostic colours).
// Geometry: POINT Grid SOP, columns x rows = lidar texture size (512 x 128 = 65,536), uv 0..1.
// MAT Samplers: sLidarPos  (lidar_positions TOP, rgba32float: xyz world TD axes, a = valid)
//               sLidarAttr (lidar_attributes TOP, rgba32float: r intensity01, g error class 0..3, b abs err m, a rel err)
// MAT Vectors : uPointSize, uRefDist, uMaxRange, uCamPos(vec3), uIntensityAmt (0..1)
uniform sampler2D sLidarPos;
uniform sampler2D sLidarAttr;
uniform float uPointSize;
uniform float uRefDist;
uniform float uMaxRange;
uniform vec3  uCamPos;
uniform float uIntensityAmt;
uniform float uGoodBelow;     // m, default 0.20 (visualisation threshold, not a sensor spec)
uniform float uLargeAbove;    // m, default 0.50
uniform float uDensity;       // 0..1 fraction of points drawn (stable per-point hash)

out float vIntensity;
flat out int   vClass;
out float vAbsErr;
out float vFade;

void main()
{
    ivec2 sz = textureSize(sLidarPos, 0);
    ivec2 px = clamp(ivec2(floor(uv[0].st * vec2(sz - 1) + 0.5)), ivec2(0), sz - 1);

    vec4 pos  = texelFetch(sLidarPos, px, 0);
    vec4 attr = texelFetch(sLidarAttr, px, 0);

    float dist = distance(pos.xyz, uCamPos);
    float h = fract(sin(dot(vec2(px), vec2(12.9898, 78.233))) * 43758.5453);
    bool visible = (pos.a > 0.5) && (dist <= uMaxRange) && (h <= uDensity);

    vec4 worldPos = TDDeform(pos.xyz);
    gl_Position = visible ? TDWorldToProj(worldPos) : vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = visible ? clamp(uPointSize * uRefDist / max(dist, 0.5), 2.0, 8.0) : 0.0;

    vIntensity = mix(1.0, 0.35 + 0.65 * attr.r, uIntensityAmt);
    // Classify at render time from the exported absolute error (attr.b, metres; < 0 = comparison unavailable)
    // so the thresholds are live controls. Class 0 orange / 1 green / 2 yellow / 3 red.
    vClass  = (attr.b < 0.0) ? 0 : ((attr.b < uGoodBelow) ? 1 : ((attr.b <= uLargeAbove) ? 2 : 3));
    vAbsErr = attr.b;
    vFade   = 1.0 - smoothstep(0.8 * uMaxRange, uMaxRange, dist);
}
