#version 300 es

precision highp float;
precision highp int;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_color;
layout(location = 3) in vec2 a_texCoord;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_viewPos;
uniform int u_renderingMode; // 0: GOURAUD, 1: PHONG

struct Material {
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform Material material;
uniform Light light;

out vec3 fragPos;
out vec3 normal;
out vec3 lightingColor; // Gouraud 셰이딩을 위한 출력 변수

void main() {
    fragPos = vec3(u_model * vec4(a_position, 1.0));
    normal = mat3(transpose(inverse(u_model))) * a_normal;
    
    // Gouraud Shading (정점 셰이더에서 조명 계산)
    if (u_renderingMode == 0) {
        // ambient
        vec3 rgb = material.diffuse;
        vec3 ambient = light.ambient * rgb;
        
        // diffuse 
        vec3 norm = normalize(normal);
        vec3 lightDir = normalize(light.position - fragPos);
        float dotNormLight = dot(norm, lightDir);
        float diff = max(dotNormLight, 0.0);
        vec3 diffuse = light.diffuse * diff * rgb;  
        
        // specular
        vec3 viewDir = normalize(u_viewPos - fragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = 0.0; 
        if (diff > 0.0) {
            spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
        }
        vec3 specular = light.specular * (spec * material.specular);
        
        // 최종 색상 계산 (정점 단위로 계산)
        lightingColor = ambient + diffuse + specular;
    }
    
    gl_Position = u_projection * u_view * vec4(fragPos, 1.0);
}