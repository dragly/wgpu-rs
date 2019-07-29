#version 450

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in vec2 a_TexCoord;
layout(location = 2) in vec4 a_InstancePos;
layout(location = 3) in vec4 a_InstanceSize;
layout(location = 0) out vec2 v_TexCoord;

layout(set = 0, binding = 0) uniform Locals {
    mat4 u_Transform;
};

void main() {
    v_TexCoord = a_TexCoord;
    gl_Position = u_Transform * vec4(a_InstanceSize.xyz * a_Pos.xyz + a_InstancePos.xyz, 1.0);
}
