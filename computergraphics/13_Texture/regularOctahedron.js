export class OregularOctahedron {
    constructor(gl, options = {}) {
        this.gl = gl;

        // VAO, VBO, EBO 생성
        this.vao = gl.createVertexArray();
        this.vbo = gl.createBuffer();
        this.ebo = gl.createBuffer();

        // 정팔면체 정점: 중심 xz평면 + 위아래 꼭짓점
        this.vertices = new Float32Array([
            0.5, 0.0, 0.5,    // v0
            0.5, 0.0, -0.5,   // v1
           -0.5, 0.0, -0.5,   // v2
           -0.5, 0.0, 0.5,    // v3
            0.0, Math.sqrt(0.5), 0.0,  // v4 (top)
            0.0, -Math.sqrt(0.5), 0.0  // v5 (bottom)
        ]);

        // (Optional) Normals: 단순화 위해 여기서는 생략하거나 나중에 필요하면 추가 가능
        // Arcball 회전만 하면 조명 없이 쓸 수 있음
        this.normals = new Float32Array([
            0, 1, 0,  // dummy normal
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0
        ]);

        // 색깔: 흰색 (필수는 아님, shader에서 color를 쓰지 않으면 무시됨)
        const baseColor = options.color || [1,1,1,1];
        this.colors = new Float32Array([
            ...baseColor, ...baseColor, ...baseColor,
            ...baseColor, ...baseColor, ...baseColor
        ]);

        // 텍스처 좌표
        this.texCoords = new Float32Array([
            1.0, 0.75,  // v0
            1.0, 0.25,  // v1
            0.0, 0.25,  // v2
            0.0, 0.75,  // v3
            0.5, 1.0,   // v4 (top)
            0.5, 0.0    // v5 (bottom)
        ]);        

        // 인덱스: 8개 삼각형 (위쪽 4개 + 아래쪽 4개)
        this.indices = new Uint16Array([
            0, 1, 4,
            1, 2, 4,
            2, 3, 4,
            3, 0, 4,
            0, 1, 5,
            1, 2, 5,
            2, 3, 5,
            3, 0, 5
        ]);
        
        

        this.initBuffers();
    }

    initBuffers() {
        const gl = this.gl;
        const vBytes = this.vertices.byteLength;
        const nBytes = this.normals.byteLength;
        const cBytes = this.colors.byteLength;
        const tBytes = this.texCoords.byteLength;
        const totalBytes = vBytes + nBytes + cBytes + tBytes;

        gl.bindVertexArray(this.vao);

        // VBO 설정 (position, normal, color, texCoord를 한 번에)
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
        gl.bufferData(gl.ARRAY_BUFFER, totalBytes, gl.STATIC_DRAW);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.vertices);
        gl.bufferSubData(gl.ARRAY_BUFFER, vBytes, this.normals);
        gl.bufferSubData(gl.ARRAY_BUFFER, vBytes + nBytes, this.colors);
        gl.bufferSubData(gl.ARRAY_BUFFER, vBytes + nBytes + cBytes, this.texCoords);

        // EBO 설정 (indices)
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.indices, gl.STATIC_DRAW);

        // vertex attribute 설정
        gl.enableVertexAttribArray(0); // position
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        gl.enableVertexAttribArray(1); // normal
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, vBytes);

        gl.enableVertexAttribArray(2); // color
        gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 0, vBytes + nBytes);

        gl.enableVertexAttribArray(3); // texCoord
        gl.vertexAttribPointer(3, 2, gl.FLOAT, false, 0, vBytes + nBytes + cBytes);

        // 정리
        gl.bindVertexArray(null);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    }

    draw(shader) {
        shader.use();
        this.gl.bindVertexArray(this.vao);
        this.gl.drawElements(this.gl.TRIANGLES, this.indices.length, this.gl.UNSIGNED_SHORT, 0);
        this.gl.bindVertexArray(null);
    }

    delete() {
        const gl = this.gl;
        gl.deleteBuffer(this.vbo);
        gl.deleteBuffer(this.ebo);
        gl.deleteVertexArray(this.vao);
    }
}


/*
export class OregularOctahedron {
    constructor(gl, options = {}) {
        this.gl = gl;

        // VAO & VBO, EBO 생성
        this.vao = gl.createVertexArray();
        this.vbo = gl.createBuffer();
        this.ebo = gl.createBuffer();

        // 정팔면체의 6개 꼭짓점
        this.vertices = new Float32Array([
             1,  0,  0,   // v0
            -1,  0,  0,   // v1
             0,  1,  0,   // v2
             0, -1,  0,   // v3
             0,  0,  1,   // v4
             0,  0, -1    // v5
        ]);

        // 각 정점의 노멀 (단위 벡터)
        this.normals = new Float32Array([
             1,  0,  0,
            -1,  0,  0,
             0,  1,  0,
             0, -1,  0,
             0,  0,  1,
             0,  0, -1
        ]);

        // 모든 정점에 단색을 지정 (흰색)
        if (options.color) {
            for (let i = 0; i < 24 * 4; i += 6) {
                this.colors[i] = options.color[0];
                this.colors[i+1] = options.color[1];
                this.colors[i+2] = options.color[2];
                this.colors[i+3] = options.color[3];
                this.colors[i+4] = options.color[4];
                this.colors[i+5] = options.color[5];
            }
        }
        else {
        const baseColor = options.color || [1,1,1,1];
        this.colors = new Float32Array([
            ...baseColor, // v0
            ...baseColor, // v1
            ...baseColor, // v2
            ...baseColor, // v3
            ...baseColor, // v4
            ...baseColor  // v5
        ]);
    }

        // 텍스처 좌표 (필요없다면 모두 0으로)
        this.texCoords = new Float32Array([
            0,0,
            0,0,
            0,0,
            0,0,
            0,0,
            0,0
        ]);

        // 8개의 삼각형 면을 구성하는 인덱스
        this.indices = new Uint16Array([
            0,4,2,
            2,4,1,
            1,4,3,
            3,4,0,
            0,2,5,
            2,1,5,
            1,3,5,
            3,0,5
        ]);

        this.initBuffers();
    }

    initBuffers() {
        const gl = this.gl;
        const vBytes = this.vertices.byteLength;
        const nBytes = this.normals.byteLength;
        const cBytes = this.colors.byteLength;
        const tBytes = this.texCoords.byteLength;
        const totalBytes = vBytes + nBytes + cBytes + tBytes;

        gl.bindVertexArray(this.vao);

        // VBO에 모든 속성 버퍼로 할당
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
        gl.bufferData(gl.ARRAY_BUFFER, totalBytes, gl.STATIC_DRAW);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0,                   this.vertices);
        gl.bufferSubData(gl.ARRAY_BUFFER, vBytes,              this.normals);
        gl.bufferSubData(gl.ARRAY_BUFFER, vBytes + nBytes,     this.colors);
        gl.bufferSubData(gl.ARRAY_BUFFER, vBytes + nBytes + cBytes, this.texCoords);

        // EBO에 인덱스 복사
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.indices, gl.STATIC_DRAW);

        // vertex attrib 설정
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);                          // position
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, vBytes);                     // normal
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 0, vBytes + nBytes);            // color
        gl.enableVertexAttribArray(3);
        gl.vertexAttribPointer(3, 2, gl.FLOAT, false, 0, vBytes + nBytes + cBytes);   // texCoord

        // 정리
        gl.bindVertexArray(null);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
    }

    draw(shader) {
        shader.use();
        this.gl.bindVertexArray(this.vao);
        // 8개 면 × 3 vertices = 24 indices
        this.gl.drawElements(this.gl.TRIANGLES, 24, this.gl.UNSIGNED_SHORT, 0);
        this.gl.bindVertexArray(null);
    }

    delete() {
        const gl = this.gl;
        gl.deleteBuffer(this.vbo);
        gl.deleteBuffer(this.ebo);
        gl.deleteVertexArray(this.vao);
    }
}
*/


/*
export class Octahedron {
    constructor(gl, options = {}) {
        this.gl = gl;

        // VAO & VBO, EBO 생성
        this.vao = gl.createVertexArray();
        this.vbo = gl.createBuffer();
        this.ebo = gl.createBuffer();

        // 정팔면체의 6개 꼭짓점
        this.vertices = new Float32Array([
             1,  0,  0,   // v0
            -1,  0,  0,   // v1
             0,  1,  0,   // v2
             0, -1,  0,   // v3
             0,  0,  1,   // v4
             0,  0, -1    // v5
        ]);

        // 각 정점의 노멀 (단위 벡터)
        this.normals = new Float32Array([
             1,  0,  0,
            -1,  0,  0,
             0,  1,  0,
             0, -1,  0,
             0,  0,  1,
             0,  0, -1
        ]);

        // 모든 정점에 단색을 지정 (흰색)
        const baseColor = options.color || [1,1,1,1];
        this.colors = new Float32Array([
            ...baseColor, // v0
            ...baseColor, // v1
            ...baseColor, // v2
            ...baseColor, // v3
            ...baseColor, // v4
            ...baseColor  // v5
        ]);

        // 텍스처 좌표 (필요없다면 모두 0으로)
        this.texCoords = new Float32Array([
            0,0,
            0,0,
            0,0,
            0,0,
            0,0,
            0,0
        ]);

        // 8개의 삼각형 면을 구성하는 인덱스
        this.indices = new Uint16Array([
            0,4,2,
            2,4,1,
            1,4,3,
            3,4,0,
            0,2,5,
            2,1,5,
            1,3,5,
            3,0,5
        ]);

        this.initBuffers();
    }

    initBuffers() {
        const gl = this.gl;
        const vBytes = this.vertices.byteLength;
        const nBytes = this.normals.byteLength;
        const cBytes = this.colors.byteLength;
        const tBytes = this.texCoords.byteLength;
        const totalBytes = vBytes + nBytes + cBytes + tBytes;

        gl.bindVertexArray(this.vao);

        // VBO에 모든 속성 버퍼로 할당
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
        gl.bufferData(gl.ARRAY_BUFFER, totalBytes, gl.STATIC_DRAW);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0,                   this.vertices);
        gl.bufferSubData(gl.ARRAY_BUFFER, vBytes,              this.normals);
        gl.bufferSubData(gl.ARRAY_BUFFER, vBytes + nBytes,     this.colors);
        gl.bufferSubData(gl.ARRAY_BUFFER, vBytes + nBytes + cBytes, this.texCoords);

        // EBO에 인덱스 복사
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ebo);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.indices, gl.STATIC_DRAW);

        // vertex attrib 설정
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);                          // position
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, vBytes);                     // normal
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 0, vBytes + nBytes);            // color
        gl.enableVertexAttribArray(3);
        gl.vertexAttribPointer(3, 2, gl.FLOAT, false, 0, vBytes + nBytes + cBytes);   // texCoord

        // 정리
        gl.bindVertexArray(null);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
    }

    draw(shader) {
        shader.use();
        this.gl.bindVertexArray(this.vao);
        // 8개 면 × 3 vertices = 24 indices
        this.gl.drawElements(this.gl.TRIANGLES, 24, this.gl.UNSIGNED_SHORT, 0);
        this.gl.bindVertexArray(null);
    }

    delete() {
        const gl = this.gl;
        gl.deleteBuffer(this.vbo);
        gl.deleteBuffer(this.ebo);
        gl.deleteVertexArray(this.vao);
    }
}

*/