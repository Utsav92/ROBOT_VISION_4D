// ROBOT VISION 4D - stereo point renderer (GLSL MAT, vertex).
// Geometry: a POINT-primitive Grid SOP whose columns/rows equal the stereo texture width/height and whose
// texture coordinates run 0..1. Every vertex samples its own 3D position/colour from float textures, so the
// whole cloud is one draw call with no per-point CPU work.
//
// MAT Samplers page : sPos   (stereo_positions TOP, rgba32float: xyz world TD axes, a = valid)
//                     sColor (stereo_colors TOP,   rgba8: camera RGB)
// MAT Vectors page  : uPointSize (float, pixels at uRefDist), uRefDist (float, m), uMaxRange (float, m),
//                     uCamPos (vec3, camera world position for range fade)
uniform sampler2D sPos;
uniform sampler2D sColor;
uniform float uPointSize;
uniform float uRefDist;
uniform float uMaxRange;
uniform vec3  uCamPos;
uniform float uDensity;      // 0..1 fraction of points drawn (stable per-point hash, so thinning does not flicker)

out vec4  vColor;
out float vFade;

void main()
{
    ivec2 sz = textureSize(sPos, 0);
    ivec2 px = ivec2(floor(uv[0].st * vec2(sz - 1) + 0.5));
    px = clamp(px, ivec2(0), sz - 1);

    vec4 pos = texelFetch(sPos, px, 0);
    vec3 rgb = texelFetch(sColor, px, 0).rgb;

    float dist = distance(pos.xyz, uCamPos);
    float h = fract(sin(dot(vec2(px), vec2(12.9898, 78.233))) * 43758.5453);
    bool visible = (pos.a > 0.5) && (dist <= uMaxRange) && (h <= uDensity);

    vec4 worldPos = TDDeform(pos.xyz);
    gl_Position = visible ? TDWorldToProj(worldPos) : vec4(2.0, 2.0, 2.0, 1.0);   // outside clip volume
    gl_PointSize = visible ? clamp(uPointSize * uRefDist / max(dist, 0.5), 1.0, 6.0) : 0.0;

    vColor = vec4(rgb, 1.0);
    vFade = 1.0 - smoothstep(0.75 * uMaxRange, uMaxRange, dist);
}
