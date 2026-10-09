// ROBOT VISION 4D - temporal ghost layer (GLSL MAT, vertex). One MAT per history layer.
// A layer is the LiDAR cloud (or just its moving-object points) of an EARLIER frame, stored in WORLD coordinates by
// preprocessing using the recorded robot pose. Because the positions are world-space, a ghost stays exactly where the
// object was, whatever the virtual camera does - this is not a 2D feedback smear.
// MAT Samplers: sHistPos (this layer's lidar position TOP: xyz world TD axes, a = valid AND (moving, in echo mode))
// MAT Vectors : uPointSize, uRefDist, uMaxRange, uCamPos(vec3), uAlpha, uAge01 (0 = now, 1 = oldest),
//               uTimeOffset(vec3, ARTISTIC displacement for Time Explosion; zero in measured-position modes)
uniform sampler2D sHistPos;
uniform float uPointSize;
uniform float uRefDist;
uniform float uMaxRange;
uniform vec3  uCamPos;
uniform float uAlpha;
uniform float uAge01;
uniform vec3  uTimeOffset;

out float vAlpha;
out float vAge;

void main()
{
    ivec2 sz = textureSize(sHistPos, 0);
    ivec2 px = clamp(ivec2(floor(uv[0].st * vec2(sz - 1) + 0.5)), ivec2(0), sz - 1);
    vec4 pos = texelFetch(sHistPos, px, 0);

    vec3 p = pos.xyz + uTimeOffset;
    float dist = distance(p, uCamPos);
    bool visible = (pos.a > 0.5) && (uAlpha > 0.004) && (dist <= uMaxRange);

    vec4 worldPos = TDDeform(p);
    gl_Position = visible ? TDWorldToProj(worldPos) : vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = visible ? clamp(uPointSize * uRefDist / max(dist, 0.5), 1.5, 6.0) : 0.0;
    vAlpha = uAlpha * (1.0 - smoothstep(0.8 * uMaxRange, uMaxRange, dist));
    vAge = uAge01;
}
