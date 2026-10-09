// ROBOT VISION 4D - temporal ghost layer (GLSL MAT, pixel). Cyan-white when recent fading to purple when old,
// semi-transparent, never the orange/green used for live measurements.
// MAT Vectors: uGlow (float 0..1)
uniform float uGlow;

in float vAlpha;
in float vAge;

layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

void main()
{
    TDCheckDiscard();
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r = length(c);
    if (r > 1.0) discard;

    vec3 recent = vec3(0.55, 0.95, 1.00);     // cyan-white
    vec3 old    = vec3(0.60, 0.30, 1.00);     // purple
    vec3 rgb = mix(recent, old, smoothstep(0.0, 1.0, vAge));
    float body = 1.0 - smoothstep(0.5, 1.0, r);
    float a = vAlpha * (body + uGlow * (1.0 - r) * 0.35);
    vec4 color = vec4(rgb, clamp(a, 0.0, 1.0));
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
