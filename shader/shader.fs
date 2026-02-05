#version 330 core
in vec3 vColor;
out vec4 FragColor;

void main()
{
    // rendre le point circulaire
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p,p);
    if (r2 > 1.0) discard;

    // pseudo éclairage "sphère"
    float z = sqrt(1.0 - r2);
    float light = 0.4 + 0.6 * z;

    FragColor = vec4(vColor * light, 1.0);
}