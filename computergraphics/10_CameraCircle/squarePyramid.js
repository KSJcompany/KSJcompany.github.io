export class squarePyramid {
    constructor(gl) {
        this.gl = gl;

        this.vao = gl.createVertexArray();
        this.vbo = gl.createBuffer();
        this.ebo = gl.createBuffer();

        // 6개의 삼각형 × 3 정점 = 총 18개의 정점 (중복 포함)
        this.vertices = new Float32Array([
            // 바닥면 2개 삼각형 (민트)
            -0.5, 0.0, -0.5,   0.5, 0.0, -0.5,   0.5, 0.0,  0.5,
            -0.5, 0.0, -0.5,   0.5, 0.0,  0.5,  -0.5, 0.0,  0.5,

            // 옆면 1 (노랑)
            -0.5, 0.0, -0.5,   0.5, 0.0, -0.5,   0.0, 1.0,  0.0,

            // 옆면 2 (핑크)
             0.5, 0.0, -0.5,   0.5, 0.0,  0.5,   0.0, 1.0,  0.0,

            // 옆면 3 (빨강)
             0.5, 0.0,  0.5,  -0.5, 0.0,  0.5,   0.0, 1.0,  0.0,

            // 옆면 4 (민트)
            -0.5, 0.0,  0.5,  -0.5, 0.0, -0.5,   0.0, 1.0,  0.0,
        ]);

        this.colors = new Float32Array([
            ...Array(6).fill([0.0, 1.0, 1.0, 1.0]).flat(), // cyan (바닥면)
            ...Array(3).fill([1.0, 1.0, 0.0, 1.0]).flat(), // yellow
            ...Array(3).fill([1.0, 0.0, 1.0, 1.0]).flat(), // magenta
            ...Array(3).fill([1.0, 0.0, 0.0, 1.0]).flat(), // red
            ...Array(3).fill([0.0, 1.0, 1.0, 1.0]).flat(), // cyan again
        ]);
        

        this.normals = new Float32Array(this.vertices.length).fill(0);  // 조명 미사용 시 기본값

        this.texCoords = new Float32Array(this.vertices.length / 3 * 2).fill(0); // 단순 초기화

        // 정점 순서대로 그리므로 0~17까지 인덱스 순서
        this.indices = new Uint16Array([
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9,10,11,
           12,13,14,
           15,16,17
        ]);

        this.initBuffers();
    }

    initBuffers() {
        const gl = this.gl;

        const vSize = this.vertices.byteLength;
        const nSize = this.normals.byteLength;
        const cSize = this.colors.byteLength;
        const tSize = this.texCoords.byteLength;
        const totalSize = vSize + nSize + cSize + tSize;

        gl.bindVertexArray(this.vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
        gl.bufferData(gl.ARRAY_BUFFER, totalSize, gl.STATIC_DRAW);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.vertices);
        gl.bufferSubData(gl.ARRAY_BUFFER, vSize, this.normals);
        gl.bufferSubData(gl.ARRAY_BUFFER, vSize + nSize, this.colors);
        gl.bufferSubData(gl.ARRAY_BUFFER, vSize + nSize + cSize, this.texCoords);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.indices, gl.STATIC_DRAW);

        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0); // position
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, vSize); // normal
        gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 0, vSize + nSize); // color
        gl.vertexAttribPointer(3, 2, gl.FLOAT, false, 0, vSize + nSize + cSize); // texCoord

        gl.enableVertexAttribArray(0);
        gl.enableVertexAttribArray(1);
        gl.enableVertexAttribArray(2);
        gl.enableVertexAttribArray(3);

        gl.bindVertexArray(null);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
    }

    draw(shader) {
        const gl = this.gl;
        shader.use();
        gl.bindVertexArray(this.vao);
        gl.drawElements(gl.TRIANGLES, this.indices.length, gl.UNSIGNED_SHORT, 0);
        gl.bindVertexArray(null);
    }

    delete() {
        const gl = this.gl;
        gl.deleteBuffer(this.vbo);
        gl.deleteBuffer(this.ebo);
        gl.deleteVertexArray(this.vao);
    }
}

