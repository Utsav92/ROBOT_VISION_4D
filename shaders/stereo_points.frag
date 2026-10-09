// ROBOT VISION 4D - stereo point renderer (GLSL MAT, pixel). Smooth round camera-coloured points.
// MAT Vectors page: uOpacity (float), uExposure (float, 1.0 = unmodified camera colour)
uniform float uOpacity;
uniform float uExposure;

in vec4  vColor;
in float vFade;

layout(location = 0) out vec4 fragColor[TD_NUM_COLOR_BUFFERS];

void main()
{
    TDCheckDiscard();
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    float edge = 1.0 - smoothstep(0.55, 1.0, r2);          // soft rim, keeps depth-test crisp enough

    vec4 color = vec4(vColor.rgb * uExposure, uOpacity * vFade * edge);
    TDAlphaTest(color.a);
    fragColor[0] = TDOutputSwizzle(color);
}
