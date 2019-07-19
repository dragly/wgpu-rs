#version 450

layout(location = 0) in vec2 v_TexCoord;
layout(location = 1) in vec4 v_Pos;
layout(location = 2) in float v_Occluder;
layout(location = 0) out vec4 o_Target;

void main() {
    float mag = length(v_TexCoord-vec2(0.5));
    vec3 tex = (v_Pos.rgb + vec3(3.0)) / 6.0;
    tex = mix(tex, vec3(0.0), mag*mag);

    if (v_Occluder > 0.0) {
        o_Target = vec4(tex.rgb, 0.5);
    } else {
        o_Target = vec4(tex.rgb, 1.0);
    }
}
