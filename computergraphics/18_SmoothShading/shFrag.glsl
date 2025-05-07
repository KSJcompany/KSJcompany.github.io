#version 300 es

precision highp float;
precision highp int;

in vec3 fragPos;  
in vec3 normal;  
in vec3 lightingColor; // Gouraud 셰이딩에서 계산된 색상
out vec4 FragColor;

struct Material {
    vec3 diffuse;       // surface's diffuse color
    vec3 specular;      // surface's specular color
    float shininess;    // specular shininess
};

struct Light {
    vec3 position;      // light position
    vec3 ambient;       // ambient strength
    vec3 diffuse;       // diffuse strength
    vec3 specular;      // specular strength
};

uniform Material material;
uniform Light light;
uniform vec3 u_viewPos;
uniform int u_renderingMode; // 0: GOURAUD, 1: PHONG

void main() {
    // Gouraud Shading (정점 셰이더에서 계산된 색상 사용)
    if (u_renderingMode == 0) {
        FragColor = vec4(lightingColor, 1.0);
        return;
    }
    
    // Phong Shading (프래그먼트 셰이더에서 조명 계산)
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
    float spec;
    if (dotNormLight > 0.0) {
        spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    }
    else spec = 0.0;
    vec3 specular = light.specular * (spec * material.specular);  
        
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}