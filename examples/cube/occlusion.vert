#version 450

layout(location = 0) in vec3 a_Pos;
layout(location = 2) in vec3 a_InstancePos;

layout(set = 0, binding = 0) uniform Locals1 {
    mat4 u_Proj;
};

//layout(set = 1, binding = 0) uniform Locals2 {
    //mat4 u_Transform;
//};

void main() {
    //gl_Position = u_Transform * vec4(a_Pos + a_InstancePos, 1.0);
    gl_Position = u_Proj * vec4(a_Pos + a_InstancePos, 1.0);
}
