// Global constants
const canvas = document.getElementById('glCanvas'); // Get the canvas element 
const gl = canvas.getContext('webgl2'); // Get the WebGL2 context

if (!gl) {
    console.error('WebGL 2 is not supported by your browser.');
}

// Set canvas size: 현재 window 전체를 canvas로 사용
canvas.width = 500;
canvas.height = 500;

// Initialize WebGL settings: viewport and clear color
gl.viewport(0, 0, canvas.width, canvas.height);
gl.clearColor(0.1, 0.2, 0.3, 1.0);

// Start rendering
render();

// Render loop
function render() {
    gl.clear(gl.COLOR_BUFFER_BIT);    
    // Draw something here

    const w = canvas.width / 2; // resize할 때마다 반으로 변경
    const h = canvas.height / 2; // resize할 때마다 반으로 변경
    
    const viewports = [
        {x : 0, y : h, color : [1.0, 0.0, 0.0, 1.0]}, // red
        {x : w, y : h, color : [0.0, 1.0, 0.0, 1.0]}, // green
        {x : 0, y : 0, color : [0.0, 0.0, 1.0, 1.0]}, // blue
        {x : w, y : 0, color : [1.0, 1.0, 0.0, 1.0]}, // yellow
    ];

    gl.clear(gl.COLOR_BUFFER_BIT);

    viewports.forEach(vp => {
        gl.viewport(vp.x, vp.y, w, h);
        gl.scissor(vp.x, vp.y, w, h);
        gl.enable(gl.SCISSOR_TEST);
        gl.clearColor(vp.color[0], vp.color[1], vp.color[2], vp.color[3]);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.disable(gl.SCISSOR_TEST);
    });
}

// Resize viewport when window size changes
window.addEventListener('resize', () => {
    const size = Math.min(window.innerWidth, window.innerHeight);
    canvas.width = size;
    canvas.height = size;
    gl.viewport(0, 0, canvas.width, canvas.height);
    render();
});


