#version 450

layout(location = 0) in vec2 v_TexCoord;
layout(location = 1) in vec4 v_Pos;
layout(location = 2) in float v_Occluder;
layout(location = 0) out vec4 o_Target;
layout(set = 0, binding = 1) uniform texture2D t_Color;
//layout(set = 0, binding = 1) uniform sampler2D t_Color;
//layout(set = 0, binding = 2) uniform samplerShadow s_Color;
//layout(set = 0, binding = 2) uniform samplerShadow s_Color;
layout(set = 0, binding = 2) uniform sampler s_Color;

void main() {
    //vec4 tex = texture(sampler2D(t_Color, s_Color), v_TexCoord);
    //vec4 tex = texture(sampler2DShadow(t_Color, s_Color), v_TexCoord);
    //vec4 tex = texture(t_Color, v_TexCoord);
    //vec4 tex = vec4(vec3(depth), 1.0);
    //float mag = length(v_TexCoord-vec2(0.5));
    //o_Target = mix(tex, vec4(0.0), mag*mag);
    //o_Target = vec4(tex.rgb, 0.1);
    //tex.rgb += v_Pos.rgb;
    vec3 tex = (v_Pos.rgb + vec3(3.0)) / 6.0;
    if (v_Occluder > 0.0) {
        o_Target = vec4(tex.rgb, 0.5);
    } else {
        o_Target = vec4(tex.rgb, 1.0);
    }
}
