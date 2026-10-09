// ROBOT VISION 4D - restrained glow + cinematic atmosphere (GLSL TOP, single pass). Bypassed in "scientific" style
// (uGlow = uVignette = 0 gives back the input colour exactly).
// Input 0: rendered 3D scene. Only pixels that are bright AND strongly saturated (LiDAR orange, box green) feed the
// glow, so the camera-coloured stereo cloud is not washed out and measurements stay readable.
// TOP Vectors page: uGlow (0..1), uRadius (pixels), uVignette (0..1), uHaze (0..1)
uniform float uGlow;
uniform float uRadius;
uniform float uVignette;
uniform float uHaze;

out vec4 fragColor;

float emissive(vec3 c)
{
    float mx = max(c.r, max(c.g, c.b));
    float mn = min(c.r, min(c.g, c.b));
    float sat = (mx - mn) / max(mx, 1e-4);
    return smoothstep(0.75, 1.0, mx) * smoothstep(0.55, 0.9, sat);
}

void main()
{
    vec2 texel = 1.0 / vec2(textureSize(sTD2DInputs[0], 0));
    vec4 base = texture(sTD2DInputs[0], vUV.st);
    vec3 rgb = base.rgb;

    if (uGlow > 0.0) {
        vec3 acc = vec3(0.0);
        float wsum = 0.0;
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                vec2 o = vec2(i, j);
                float w = exp(-dot(o, o) / 3.0);
                vec3 s = texture(sTD2DInputs[0], vUV.st + o * texel * uRadius).rgb;
                acc += s * emissive(s) * w;
                wsum += w;
            }
        }
        // soft-limited (Reinhard) so a dense cluster of bright points cannot blow out to flat colour
        vec3 g = acc / wsum * uGlow * 2.0;
        rgb += 0.55 * g / (1.0 + g);
    }

    // atmosphere: a faint blue-charcoal haze toward the top of frame and a soft vignette
    float r = length(vUV.st - vec2(0.5)) * 1.41421356;
    rgb += uHaze * vec3(0.010, 0.016, 0.030) * smoothstep(0.2, 1.0, vUV.t);
    rgb *= 1.0 - uVignette * smoothstep(0.55, 1.1, r);

    fragColor = TDOutputSwizzle(vec4(rgb, base.a));
}
