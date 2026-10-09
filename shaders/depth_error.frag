// ROBOT VISION 4D - stereo-vs-LiDAR depth disagreement colouring (GLSL MAT, pixel; pairs with lidar_points.vert).
// Class (from depth_comparison.py): 0 unavailable -> orange, 1 good -> green, 2 moderate -> yellow, 3 large -> red.
// Colours are diagnostic, not a claim that the stereo estimate is wrong (see README: occlusion, sync, reflectivity).
// MAT Vectors: uOpacity (float)
uniform float uOpacity;

in float vIntensity;
flat in int   vClass;
in float vAbsErr;
in float vFade;

layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

vec3 classColor(int k)
{
    if (k == 1) return vec3(0.10, 1.00, 0.35);   // green  : close agreement
    if (k == 2) return vec3(1.00, 0.88, 0.10);   // yellow : moderate disagreement
    if (k == 3) return vec3(1.00, 0.12, 0.10);   // red    : large disagreement
    return vec3(1.0, 0.4588, 0.0);               // orange : comparison unavailable
}

void main()
{
    TDCheckDiscard();
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r = length(c);
    if (r > 1.0) discard;
    float disc = 1.0 - smoothstep(0.6, 1.0, r);
    vec4 color = vec4(classColor(vClass) * vIntensity, disc * uOpacity * vFade);
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
