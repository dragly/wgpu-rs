#version 450

layout(location = 0) in vec2 v_TexCoord;
layout(location = 0) out vec4 o_Target;

void main() {
    float mag = length(v_TexCoord-vec2(0.5));
    o_Target = vec4(mag*mag, 0.0, 0.0, 1.0);
}
