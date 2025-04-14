import { resizeAspectRatio, setupText, updateText, Axes } from '../util/util.js';
import { Shader, readShaderFile } from '../util/shader.js';
import { squarePyramid } from './squarePyramid.js';

const canvas = document.getElementById('glCanvas');
const gl = canvas.getContext('webgl2');
let shader;
let startTime;
let lastFrameTime;

let isInitialized = false;

let viewMatrix = mat4.create();
let projMatrix = mat4.create();
let modelMatrix = mat4.create(); 

const cameraCircleRadius = 3.0;
const cameraCircleSpeedXZ = 90.0;       // deg/sec
const cameraVerticalSpeedY = 45.0;      // deg/sec

const pyramid = new squarePyramid(gl);
const axes = new Axes(gl, 1.8);

document.addEventListener('DOMContentLoaded', () => {
    if (isInitialized) {
        console.log("Already initialized");
        return;
    }

    main().then(success => {
        if (!success) {
            console.log('program terminated');
            return;
        }
        isInitialized = true;
    }).catch(error => {
        console.error('program terminated with error:', error);
    });
});

function initWebGL() {
    if (!gl) {
        console.error('WebGL 2 is not supported by your browser.');
        return false;
    }

    canvas.width = 700;
    canvas.height = 700;
    resizeAspectRatio(gl, canvas);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.7, 0.8, 0.9, 1.0);
    
    return true;
}

async function initShader() {
    const vertexShaderSource = await readShaderFile('shVert.glsl');
    const fragmentShaderSource = await readShaderFile('shFrag.glsl');
    return new Shader(gl, vertexShaderSource, fragmentShaderSource);
}

function render() {
    const currentTime = Date.now();
    const deltaTime = (currentTime - lastFrameTime) / 1000.0;
    const elapsedTime = (currentTime - startTime) / 1000.0;
    lastFrameTime = currentTime;

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);

    mat4.identity(modelMatrix); // 사각뿔은 고정

    // 카메라 위치 계산
    const angleXZ = glMatrix.toRadian(90.0 * elapsedTime);
    const angleY = glMatrix.toRadian(45.0 * elapsedTime);

    const camX = 3.0 * Math.sin(angleXZ);
    const camZ = 3.0 * Math.cos(angleXZ);
    const camY = 5.0 + 5.0 * Math.sin(angleY);  // y = 0 ~ 10 반복

    mat4.lookAt(
        viewMatrix,
        vec3.fromValues(camX, camY, camZ),   // 카메라 위치
        vec3.fromValues(0, 0, 0),            // 원점 바라봄
        vec3.fromValues(0, 1, 0)             // up 방향
    );

    shader.use();
    shader.setMat4('u_model', modelMatrix);
    shader.setMat4('u_view', viewMatrix);
    shader.setMat4('u_projection', projMatrix);
    pyramid.draw(shader);
    axes.draw(viewMatrix, projMatrix);

    requestAnimationFrame(render);
}


async function main() {
    try {
        if (!initWebGL()) throw new Error('WebGL initialization failed');

        shader = await initShader();

        mat4.perspective(
            projMatrix,
            glMatrix.toRadian(60),               // FOV
            canvas.width / canvas.height,        // Aspect ratio
            0.1,                                  // near
            100.0                                 // far
        );

        startTime = lastFrameTime = Date.now();
        requestAnimationFrame(render);
        return true;
    } catch (error) {
        console.error('Failed to initialize program:', error);
        alert('Failed to initialize program');
        return false;
    }
}


