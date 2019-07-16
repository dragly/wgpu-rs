#version 450

layout(location = 0) in vec2 v_TexCoord;
layout(location = 0) out vec4 o_Target;
layout(set = 0, binding = 1) uniform texture2D t_Color;
//layout(set = 0, binding = 1) uniform sampler2D t_Color;
//layout(set = 0, binding = 2) uniform samplerShadow s_Color;
//layout(set = 0, binding = 2) uniform samplerShadow s_Color;
layout(set = 0, binding = 2) uniform sampler s_Color;

void main() {
    vec4 tex = texture(sampler2D(t_Color, s_Color), v_TexCoord);
    //vec4 tex = texture(sampler2DShadow(t_Color, s_Color), v_TexCoord);
    //vec4 tex = texture(t_Color, v_TexCoord);
    //vec4 tex = vec4(vec3(depth), 1.0);
    float mag = length(v_TexCoord-vec2(0.5));
    o_Target = mix(tex, vec4(0.0), mag*mag);
}
