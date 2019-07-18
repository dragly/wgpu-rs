#version 450

layout(location = 0) in vec3 a_Pos;
layout(location = 1) in vec2 a_TexCoord;
layout(location = 2) in vec4 a_InstancePos;
layout(location = 3) in vec4 a_InstanceSize;
layout(location = 0) out vec2 v_TexCoord;
layout(location = 1) out vec4 v_Pos;
layout(location = 2) out float v_Occluder;

layout(set = 0, binding = 0) uniform Locals {
    mat4 u_Transform;
};

void main() {
    v_TexCoord = a_TexCoord;
    gl_Position = u_Transform * vec4(a_Pos.xyz * a_InstanceSize.xyz + a_InstancePos.xyz, 1.0);
    v_Occluder = a_InstanceSize.x > 3.0 ? 1 : 0;
    v_Pos = a_InstancePos;
}
