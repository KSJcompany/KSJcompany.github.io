/*-------------------------------------------------------------------------
07_LineSegments.js

1) 처음 실행했을 때, canvas의 크기는 700 x 700 이어야 합니다.
2) 먼저 중심점을 왼쪽 마우스버튼 클릭한 채로 dragging하여 반지름을 늘리고 줄이다가 마우스버튼을 놓아 circle을 입력합니다.
Circle은 NDC의 범위가 넘어갈 수도 있으며, 그 경우 NDC의 범위 안에 있는 부분만 그려지게 됩니다. Circle의 정보가 info 첫번째 line에 나타납니다.
3) 두번째로 line segment를 07_LineSegments 프로그램과 같이 입력합니다. Line segment의 정보가 info 두번째 line에 나타납니다.
4) Line segment 입력이 끝나자 마자 intersection point를 계산하며, intersection이 있는 경우, 아래 그림과 같이 intersection point의 개수와 coordinates가 info 세번째 line에 표시됩니다.
5) Intersection point의 size는 10.0으로 하며, vertex shader의 main() 안에서 gl_PointSize = 10.0; 과 같이 크기를 정의합니다.

---------------------------------------------------------------------------*/
import { resizeAspectRatio, setupText, updateText, Axes } from '../util/util.js';
import { Shader, readShaderFile } from '../util/shader.js';

// Global variables
let isInitialized = false; // global variable로 event listener가 등록되었는지 확인
const canvas = document.getElementById('glCanvas');
const gl = canvas.getContext('webgl2');
let shader;
let vao;
let positionBuffer;
let isDrawing = false;
let startPoint = null;
let tempEndPoint = null;

let circleCenter = null;
let circleRadius = 0;
let circleDefined = false;

let lineStart = null;
let lineEnd = null;
let lines = [];

let intersectionPoints = []; // 교차점 좌표들 저장

let textOverlay;
let textOverlay2;
let textOverlay3;
let axes = new Axes(gl, 0.85);

// DOMContentLoaded event
// 1) 모든 HTML 문서가 완전히 load되고 parsing된 후 발생
// 2) 모든 resource (images, css, js 등) 가 완전히 load된 후 발생
// 3) 모든 DOM 요소가 생성된 후 발생
// DOM: Document Object Model로 HTML의 tree 구조로 표현되는 object model 
// 모든 code를 이 listener 안에 넣는 것은 mouse click event를 원활하게 처리하기 위해서임

/* 수정전 코드
function resizeCanvas() {
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;
    console.log(`client: ${displayWidth}x${displayHeight}, buffer: ${canvas.width}x${canvas.height}`);

    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        gl.viewport(0, 0, canvas.width, canvas.height);
    }
}
*/
function resizeCanvas() {
    const displayWidth = canvas.clientWidth; // canvas의 현재 표시 너비
    const displayHeight = canvas.clientHeight; // canvas의 현재 표시 높이

    console.log(`client: ${displayWidth}x${displayHeight}, buffer: ${canvas.width}x${canvas.height}`);

    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;

        // 스타일 크기 재설정
        canvas.style.width = `${displayWidth}px`;
        canvas.style.height = `${displayHeight}px`;

        gl.viewport(0, 0, canvas.width, canvas.height);
    }
}



// mouse 쓸 때 main call 방법
document.addEventListener('DOMContentLoaded', () => {
    if (isInitialized) {
        console.log("Already initialized");
        return;
    }

    main().then(success => { // call main function
        if (!success) {
            console.log('프로그램을 종료합니다.');
            return;
        }
        isInitialized = true;
    }).catch(error => {
        console.error('프로그램 실행 중 오류 발생:', error);
    });
});

function initWebGL() {
    if (!gl) {
        console.error('WebGL 2 is not supported by your browser.');
        return false;
    }

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.7, 0.8, 0.9, 1.0);
    
    return true;
}

function setupCanvas() {
    canvas.style.width = '700px';
    canvas.style.height = '700px';

    canvas.width = 700;
    canvas.height = 700;

    resizeAspectRatio(gl, canvas);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.1, 0.2, 0.3, 1.0);
}


function setupBuffers(shader) {
    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

    shader.setAttribPointer('a_position', 2, gl.FLOAT, false, 0, 0);

    gl.bindVertexArray(null);
}

// 좌표 변환 함수: 캔버스 좌표를 WebGL 좌표로 변환
// 캔버스 좌표: 캔버스 좌측 상단이 (0, 0), 우측 하단이 (canvas.width, canvas.height)
// WebGL 좌표 (NDC): 캔버스 좌측 상단이 (-1, 1), 우측 하단이 (1, -1)
function convertToWebGLCoordinates(x, y) {
    return [
        (x / canvas.width) * 2 - 1,
        -((y / canvas.height) * 2 - 1)
    ];
}

// 임시 원 그리기
function calculateCirclePoints(center, radius, segment = 100) {
    const circle = [];
    for (let i = 0; i < segment; i++) {
        const angle = (i / segment) * 2 * Math.PI;
        const x = center[0] + radius * Math.cos(angle);
        const y = center[1] + radius * Math.sin(angle);
        circle.push(x, y);
    }
    return circle;
}

// 원 정보 저장 함수
function saveCircleInfo(center, radius, segment = 100) {
    const circle = [];
    for (let i = 0; i < segment; i++) {
        const angle = (i / segment) * 2 * Math.PI;
        const x = center[0] + radius * Math.cos(angle);
        const y = center[1] + radius * Math.sin(angle);
        circle.push(x, y);
    }
    return circle;
}

function getCircleSegmentIntersections(center, radius, p1, p2) {
    // 직선-원 교차점 공식 기반으로 계산
    const dx = p2[0] - p1[0];
    const dy = p2[1] - p1[1];
    const fx = p1[0] - center[0];
    const fy = p1[1] - center[1];

    const a = dx * dx + dy * dy;
    const b = 2 * (fx * dx + fy * dy);
    const c = fx * fx + fy * fy - radius * radius;

    const discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return [];

    const sqrtDisc = Math.sqrt(discriminant);
    const t1 = (-b - sqrtDisc) / (2 * a);
    const t2 = (-b + sqrtDisc) / (2 * a);
    const points = [];

    if (t1 >= 0 && t1 <= 1) {
        points.push([p1[0] + t1 * dx, p1[1] + t1 * dy]);
    }
    if (t2 >= 0 && t2 <= 1) {
        points.push([p1[0] + t2 * dx, p1[1] + t2 * dy]);
    }

    return points;
}

/* 
    browser window
    +----------------------------------------+
    | toolbar, address bar, etc.             |
    +----------------------------------------+
    | browser viewport (컨텐츠 표시 영역)       | 
    | +------------------------------------+ |
    | |                                    | |
    | |    canvas                          | |
    | |    +----------------+              | |
    | |    |                |              | |
    | |    |      *         |              | |
    | |    |                |              | |
    | |    +----------------+              | |
    | |                                    | |
    | +------------------------------------+ |
    +----------------------------------------+

    *: mouse click position

    event.clientX = browser viewport 왼쪽 경계에서 마우스 클릭 위치까지의 거리
    event.clientY = browser viewport 상단 경계에서 마우스 클릭 위치까지의 거리
    rect.left = browser viewport 왼쪽 경계에서 canvas 왼쪽 경계까지의 거리
    rect.top = browser viewport 상단 경계에서 canvas 상단 경계까지의 거리

    x = event.clientX - rect.left  // canvas 내에서의 클릭 x 좌표
    y = event.clientY - rect.top   // canvas 내에서의 클릭 y 좌표
*/

function setupMouseEvents() {
    function handleMouseDown(event) {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const [glX, glY] = convertToWebGLCoordinates(x, y);
    
        isDrawing = true;
    
        if (!circleDefined) {
            // 초기화
            lines = [];
            intersectionPoints = [];
            updateText(textOverlay, "", 1);
            updateText(textOverlay2, "", 2);
            updateText(textOverlay3, "", 3);
            updateText(canvas, "", 3);

            circleCenter = [glX, glY]; // 원 중심 설정
        } 
        else {
            lineStart = [glX, glY]; // 선분 시작점 설정
        }
    }
    

    function handleMouseMove(event) {
        if (!isDrawing) return;
    
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        let [glX, glY] = convertToWebGLCoordinates(x, y);
        tempEndPoint = [glX, glY];
    
        render();
    }
    


    function handleMouseUp() {
        if (!isDrawing || !tempEndPoint) return;
    
        if (!circleDefined) {
            // 원 그리기 단계
            circleCenter = [...circleCenter]; // 이미 mousedown에서 지정됨
            const dx = tempEndPoint[0] - circleCenter[0];
            const dy = tempEndPoint[1] - circleCenter[1];
            circleRadius = Math.sqrt(dx * dx + dy * dy);
    
            const circlePoints = calculateCirclePoints(circleCenter, circleRadius);
            lines.push(circlePoints);
            updateText(textOverlay, `Circle center: (${circleCenter[0].toFixed(2)}, ${circleCenter[1].toFixed(2)})  radius = ${circleRadius.toFixed(2)}`);
    
            circleDefined = true; // 원이 정의됨
        } 
        else {
            // 선분 그리기 단계
            lineEnd = [...tempEndPoint];
            lines.push([...lineStart, ...tempEndPoint]);
    
            updateText(textOverlay2, `Line segment: (${lineStart[0].toFixed(2)}, ${lineStart[1].toFixed(2)}) ~ (${lineEnd[0].toFixed(2)}, ${lineEnd[1].toFixed(2)})`);
    
            // 교차점 계산
            const intersections = getCircleSegmentIntersections(circleCenter, circleRadius, lineStart, lineEnd);

            intersectionPoints = intersections; // 전역 배열에 저장

            if (intersections.length > 0) {
                const msg = `Intersection Points: ${intersections.length}\n` +
                    intersections.map((p, i) => `Point ${i + 1}: (${p[0].toFixed(2)}, ${p[1].toFixed(2)})`).join('\n');
                updateText(textOverlay3, msg);
            }     
            else {
                updateText(textOverlay3, "No intersection");
            }

            circleDefined = false;
            circleCenter = null;
            circleRadius = 0;

        }
    
        // 공통 마무리
        isDrawing = false;
        tempEndPoint = null;
        render();
    }
    

    canvas.addEventListener("mousedown", handleMouseDown);
    canvas.addEventListener("mousemove", handleMouseMove);
    canvas.addEventListener("mouseup", handleMouseUp);
    }

function render() {
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);

    shader.use();
    
    // 저장된 선들 그리기
    let num = 0;
    for (let line of lines) {
        if (num === 0) { // 원인 경우
            shader.setVec4("u_color", [0.5, 0.0, 0.5, 1.0]); // 보라색
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(line), gl.STATIC_DRAW);
            gl.bindVertexArray(vao);
            gl.drawArrays(gl.LINE_LOOP, 0, line.length / 2);
        } 
        else { // 선분인 경우
            shader.setVec4("u_color", [1.0, 0.0, 1.0, 1.0]); // 마젠타
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(line), gl.STATIC_DRAW);
            gl.bindVertexArray(vao);
            gl.drawArrays(gl.LINES, 0, 2);
        }
        num++;
    }

    if (isDrawing && tempEndPoint) {
        if (!circleDefined && circleCenter) {
            const dx = tempEndPoint[0] - circleCenter[0];
            const dy = tempEndPoint[1] - circleCenter[1];
            const radius = Math.sqrt(dx * dx + dy * dy);
            const tempCircle = calculateCirclePoints(circleCenter, radius);
            shader.setVec4("u_color", [0.6, 0.6, 0.6, 1.0]);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(tempCircle), gl.STATIC_DRAW);
            gl.bindVertexArray(vao);
            gl.drawArrays(gl.LINE_LOOP, 0, tempCircle.length / 2);
        } 
        else if (lineStart) {
            shader.setVec4("u_color", [0.5, 0.5, 0.5, 1.0]);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([...lineStart, ...tempEndPoint]), gl.STATIC_DRAW);
            gl.bindVertexArray(vao);
            gl.drawArrays(gl.LINES, 0, 2);
        }
    }
    
    if (intersectionPoints.length > 0) {
        shader.setVec4("u_color", [1.0, 1.0, 0.0, 1.0]); // 노란 점
        const flatPoints = intersectionPoints.flat(); // [[x, y], [x, y]] -> [x, y, x, y]
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(flatPoints), gl.STATIC_DRAW);
        gl.bindVertexArray(vao);
        gl.drawArrays(gl.POINTS, 0, intersectionPoints.length);
    }

    // axes 그리기
    axes.draw(mat4.create(), mat4.create());
}

async function initShader() {
    const vertexShaderSource = await readShaderFile('shVert.glsl');
    const fragmentShaderSource = await readShaderFile('shFrag.glsl');
    return new Shader(gl, vertexShaderSource, fragmentShaderSource);
}

async function main() {
    window.addEventListener('resize', () => {
        resizeCanvas();
        render();
    });
    
    try {
        if (!initWebGL()) {
            throw new Error('WebGL 초기화 실패');
        }

        // 셰이더 초기화
        shader = await initShader();
        
        // 나머지 초기화
        setupCanvas();
        setupBuffers(shader);
        shader.use();

        // 텍스트 초기화
        textOverlay = setupText(canvas, "", 1);
        textOverlay2 = setupText(canvas, "", 2);
        textOverlay3 = setupText(canvas, "", 3);

        // 마우스 이벤트 설정
        setupMouseEvents();
        
        // 초기 렌더링
        render();

        return true;
    } catch (error) {
        console.error('Failed to initialize program:', error);
        alert('프로그램 초기화에 실패했습니다.');
        return false;
    }
}
