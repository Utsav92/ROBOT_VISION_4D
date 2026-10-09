// ROBOT VISION 4D - LiDAR points as bright orange (#FF7500) glowing discs (GLSL MAT, pixel).
// MAT Vectors: uOpacity (float), uGlow (float 0..1: soft halo strength baked into the sprite)
uniform float uOpacity;
uniform float uGlow;

in float vIntensity;
flat in int   vClass;
in float vAbsErr;
in float vFade;

layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

const vec3 LIDAR_ORANGE = vec3(1.0, 0.4588, 0.0);          // #FF7500

void main()
{
    TDCheckDiscard();
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r = length(c);
    if (r > 1.0) discard;

    float core = 1.0 - smoothstep(0.35, 0.65, r);           // hot centre
    float halo = (1.0 - smoothstep(0.0, 1.0, r)) * uGlow;   // subtle glow, never swamps stereo colours
    float a = clamp(core + halo * 0.5, 0.0, 1.0) * uOpacity * vFade;

    vec3 rgb = LIDAR_ORANGE * vIntensity * (1.0 + 0.6 * core);
    vec4 color = vec4(rgb, a);
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
