import { resizeAspectRatio, setupText, updateText, Axes } from '../util/util.js';
import { Shader, readShaderFile } from '../util/shader.js';

let isInitialized = false;
const canvas = document.getElementById('glCanvas');
const gl = canvas.getContext('webgl2');
let shader;
let vao;
let axes;

// 전역 시간 변수 (초)
let time = 0;
let lastTime = 0;

document.addEventListener('DOMContentLoaded', () => {
    if (isInitialized) return;
    main().then(success => {
        if (success) {
            isInitialized = true;
            requestAnimationFrame(animate);
        } else {
            console.log('프로그램 초기화 실패');
        }
    }).catch(error => {
        console.error('프로그램 실행 중 오류 발생:', error);
    });
});

function initWebGL() {
    if (!gl) {
        console.error('WebGL 2를 지원하지 않습니다.');
        return false;
    }
    canvas.width = 700;
    canvas.height = 700;
    resizeAspectRatio(gl, canvas);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.2, 0.3, 0.4, 1.0);
    return true;
}

function setupBuffers() {
    const vertices = new Float32Array([
        -0.5,  0.5, 0.0,
        -0.5, -0.5, 0.0,
         0.5, -0.5, 0.0,
         0.5,  0.5, 0.0
    ]);

    // 이 버퍼의 색상은 이후에 constant attribute로 덮어씁니다.
    const colors = new Float32Array([
        1, 0, 0, 1,  
        0, 1, 0, 1,  
        0, 0, 1, 1,  
        1, 1, 0, 1   
    ]);

    const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);

    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    // a_position (location 0)
    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    // a_color (location 1) – 나중에 constant attribute로 설정할 예정입니다.
    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);

    // 인덱스 버퍼
    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    gl.bindVertexArray(null);
}

function render() {
    gl.clear(gl.COLOR_BUFFER_BIT);
    axes.draw(mat4.create(), mat4.create());

    shader.use();
    gl.bindVertexArray(vao);

    // 계산할 각도 (시간에 따라)
    const angle1 = (Math.PI / 4) * time;       // model1 자전: 45°/s
    const angle2_self = Math.PI * time;          // model2 자전: 180°/s
    const angle2_orbit = (Math.PI / 6) * time;     // model2 공전: 30°/s
    const angle3_self = Math.PI * time;          // model3 자전: 180°/s
    const angle3_orbit = 2 * Math.PI * time;       // model3 공전: 360°/s

    // ── model1 (중심, red, edge length 0.2, 자전) ──
    let model1 = mat4.create();
    mat4.rotate(model1, model1, angle1, [0, 0, 1]);
    mat4.scale(model1, model1, [0.2, 0.2, 1]);  // edge length 0.2
    // 색상 red
    gl.disableVertexAttribArray(1);
    gl.vertexAttrib4f(1, 1.0, 0.0, 0.0, 1.0);
    shader.setMat4("u_model", model1);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);

    // ── model2 (중심 기준 공전, cyan, edge length 0.1) ──
    let model2 = mat4.create();
    // 공전: 중심에서 0.7의 반지름을 따라 공전 (각도 angle2_orbit)
    mat4.translate(model2, model2, [0.7 * Math.cos(angle2_orbit), 0.7 * Math.sin(angle2_orbit), 0]);
    // 자전: model2 자체 회전 (각도 angle2_self)
    mat4.rotate(model2, model2, angle2_self, [0, 0, 1]);
    // 스케일: edge length 0.1
    mat4.scale(model2, model2, [0.1, 0.1, 1]);
    gl.vertexAttrib4f(1, 0.0, 1.0, 1.0, 1.0); // cyan
    shader.setMat4("u_model", model2);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);

    // ── model3 (model2 기준 공전, yellow, edge length 0.05) ──
    // model2의 스케일이 0.1이므로, model3의 부모 기준 offset을 0.2/0.1 = 2.0로 합니다.
    let model3 = mat4.create();
    
    // M_orbit: model3의 공전 변환 (모델2 기준으로 2.0의 거리, 각도 angle3_orbit)
    // 회전 먼저 적용하고 그 후 translation
    let m_orbit = mat4.create();
    mat4.rotate(m_orbit, m_orbit, angle3_orbit, [0, 0, 1]);
    mat4.translate(m_orbit, m_orbit, [2.0, 0, 0]);
    
    // M_self: model3의 자전 및 로컬 스케일 (자전 angle3_self, 스케일 0.5 → 0.1 * 0.5 = 0.05)
    let m_self = mat4.create();
    mat4.rotate(m_self, m_self, angle3_self, [0, 0, 1]);
    mat4.scale(m_self, m_self, [0.5, 0.5, 1]);

    // model3 = model2 * M_orbit * M_self
    let m_local = mat4.create();
    mat4.multiply(m_local, m_orbit, m_self);
    mat4.multiply(model3, model2, m_local);
    
    gl.vertexAttrib4f(1, 1.0, 1.0, 0.0, 1.0); // yellow
    shader.setMat4("u_model", model3);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);


    gl.bindVertexArray(null);
    
}

function animate(currentTime) {
    if (!lastTime) lastTime = currentTime;
    const deltaTime = (currentTime - lastTime) / 1000;
    lastTime = currentTime;
    time += deltaTime;
    render();
    requestAnimationFrame(animate);
}

async function initShader() {
    const vertexShaderSource = await readShaderFile('shVert.glsl');
    const fragmentShaderSource = await readShaderFile('shFrag.glsl');
    shader = new Shader(gl, vertexShaderSource, fragmentShaderSource);
}

async function main() {
    try {
        if (!initWebGL()) throw new Error('WebGL 초기화 실패');
        await initShader();
        setupBuffers();
        axes = new Axes(gl, 1.0);
        return true;
    } catch (error) {
        console.error('프로그램 초기화 실패:', error);
        alert('프로그램 초기화에 실패했습니다.');
        return false;
    }
}
