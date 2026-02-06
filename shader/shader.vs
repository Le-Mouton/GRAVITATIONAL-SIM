#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aColor;
layout (location=2) in float aRadius;

out vec3 vColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3  uCameraPos;
uniform float uViewportH;
uniform float uFovY;

void main()
{
    vec4 worldPos4 = model * vec4(aPos, 1.0);
    vec3 worldPos  = worldPos4.xyz;

    gl_Position = projection * view * worldPos4;
    vColor = aColor;

    float d = length(uCameraPos - worldPos);
    d = max(d, 1e-3);

    float pixelsPerWorld = uViewportH / (2.0 * tan(uFovY * 0.5));
    float sizePx = (aRadius / d) * pixelsPerWorld * 2.0;

    gl_PointSize = clamp(sizePx, 1.0, 2048.0);
}