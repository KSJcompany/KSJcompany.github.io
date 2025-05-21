// main three.module.js library
import * as THREE from 'three';  
// addons: OrbitControls (jsm/controls), Stats (jsm/libs), GUI (jsm/libs)
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'three/addons/libs/stats.module.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';

const scene = new THREE.Scene();
scene.backgroundColor = 0x000000;
// fog 효과, camera로 부터 0.0025 거리에서는 fog가 없고
// 거리 50에서는 어떤 object든 fog (white color)에 둘러싸여 보이지 않음
//scene.fog = new THREE.Fog(0xffffff, 0.0025, 50); 

// Perspective camera: fov, aspect ratio, near, far
let aspectRatio = window.innerWidth / window.innerHeight;
let camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.1, 1000);
camera.position.x = 0;
camera.position.y = 70;
camera.position.z = 120;
scene.add(camera);

// set camera position: camera.position.set(-3, 8, 2) 가 더 많이 사용됨 (약간 빠름))
const orthographicCamera = new THREE.OrthographicCamera(
    -window.innerWidth / 5,
    window.innerWidth / 5,
    window.innerHeight / 5,
    -window.innerHeight / 5,
    0.1,
    1000
);

orthographicCamera.position.copy(camera.position);

// add camera to the scene
scene.add(orthographicCamera);

let activeCamera = camera;

// setup the renderer and attch to canvas
// antialias = true: 렌더링 결과가 부드러워짐
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.outputColorSpace = THREE.SRGBColorSpace;


// outputColorSpace의 종류
// sRGBColorSpace: 보통 monitor에서 보이는 color로, 어두운 부분을 약간 밝게 보이게 Gamma correction을 함
// sRGBColorSpace는 PBR (Physically Based Rendering), HDR(High Dynamic Range)에서는 필수적으로 사용함
// LinearColorSpace: 모든 색상을 선형으로 보이게 함
renderer.outputColorSpace = THREE.SRGBColorSpace;

renderer.shadowMap.enabled = true; // scene에서 shadow를 보이게 할 겁니다. 

// shadowMap의 종류
// BasicShadowMap: 가장 기본적인 shadow map, 쉽고 빠르지만 부드럽지 않음
// PCFShadowMap (default): Percentage-Closer Filtering, 주변의 색상을 평균내서 부드럽게 보이게 함
// PCFSoftShadowMap: 더 부드럽게 보이게 함
// VSMShadowMap: Variance Shadow Map, 더 자연스러운 블러 효과, GPU에서 더 많은 연산 필요
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

// 현재 열린 browser window의 width와 height에 맞게 renderer의 size를 설정
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000);
// attach renderer to the body of the html page
document.body.appendChild(renderer.domElement);

// add Stats: 현재 FPS를 보여줌으로써 rendering 속도 표시
const stats = new Stats();
// attach Stats to the body of the html page
document.body.appendChild(stats.dom);

// add OrbitControls: arcball-like camera control
const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.enableDamping = true;
orbitControls.dampingFactor = 0.05;

// add GUI: 간단한 user interface를 제작 가능
// 사용법은 https://lil-gui.georgealways.com/ 
// http://yoonbumtae.com/?p=942 참고

const planets = [];

const gui = new GUI();
const cameraFolder = gui.addFolder('Camera');
const cameraProps = {
    cameraType: 'Perspective'
};

cameraFolder.add(cameraProps, 'cameraType', ['Perspective', 'Orthographic']).onChange((value) => {
    if (value === 'Perspective') {
        activeCamera = camera;
        orbitControls.object = camera;
    } else {
        activeCamera = orthographicCamera;
        orthographicCamera.position.copy(camera.position);
        orthographicCamera.rotation.copy(camera.rotation);
        orbitControls.object = orthographicCamera;
    }
    orbitControls.update();
});

// listen to the resize events
window.addEventListener('resize', onResize, false);
function onResize() { // resize handler
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    
    orthographicCamera.left = -window.innerWidth / 5;
    orthographicCamera.right = window.innerWidth / 5;
    orthographicCamera.top = window.innerHeight / 5;
    orthographicCamera.bottom = -window.innerHeight / 5;
    orthographicCamera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);
}

// add ambient light
const ambientLight = new THREE.AmbientLight(0x888888);
scene.add(ambientLight);

const sunLight = new THREE.PointLight(0xffffff, 1.5, 1000);
sunLight.position.set(0, 0, 0);
scene.add(sunLight);

const textureLoader = new THREE.TextureLoader();

const sunGeometry = new THREE.SphereGeometry(10, 32, 32);
const sunMaterial = new THREE.MeshBasicMaterial({ 
    color: 0xffff00
});

const sunGlow = new THREE.PointLight(0xffff00, 1.5, 300);
scene.add(sunGlow);

const sun = new THREE.Mesh(sunGeometry, sunMaterial);
scene.add(sun);

function createPlanet(name, radius, distance, color, rotationSpeed, orbitSpeed, texturePath) {
    const planetGroup = new THREE.Group();
    scene.add(planetGroup);
    
    // Planet orbit line
    const orbitGeometry = new THREE.RingGeometry(distance - 0.1, distance + 0.1, 64);
    const orbitMaterial = new THREE.MeshBasicMaterial({ 
        color: 0x444444,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.3
    });
    const orbit = new THREE.Mesh(orbitGeometry, orbitMaterial);
    orbit.rotation.x = Math.PI / 2;
    scene.add(orbit);
    
    // Planet geometry
    const planetGeometry = new THREE.SphereGeometry(radius, 32, 32);
    
    // Load texture
    let planetMaterial;
    if (texturePath) {
        const texture = textureLoader.load(texturePath);
        planetMaterial = new THREE.MeshStandardMaterial({
            map: texture,
            roughness: 0.8,
            metalness: 0.2
        });
    } else {
        planetMaterial = new THREE.MeshStandardMaterial({ color: color });
    }
    
    const planetMesh = new THREE.Mesh(planetGeometry, planetMaterial);
    planetGroup.add(planetMesh);
    
    // Set initial position
    planetMesh.position.x = distance;
    
    // Create GUI folder for this planet
    const planetFolder = gui.addFolder(name);
    const planetProps = {
        rotationSpeed: rotationSpeed,
        orbitSpeed: orbitSpeed,
        distance: distance
    };
    
    planetFolder.add(planetProps, 'rotationSpeed', 0, 0.05, 0.001).name('Rotation Speed');
    planetFolder.add(planetProps, 'orbitSpeed', 0, 0.05, 0.001).name('Orbit Speed');
    planetFolder.add(planetProps, 'distance', distance * 0.5, distance * 1.5, 1).onChange((value) => {
        orbit.geometry.dispose();
        orbit.geometry = new THREE.RingGeometry(value - 0.1, value + 0.1, 64);
    }).name('Distance');
    
    // Add to planets array
    planets.push({
        mesh: planetMesh,
        group: planetGroup,
        props: planetProps
    });
    
    return planetGroup;
}

createPlanet('Mercury', 1.5, 20, 0xa6a6a6, 0.02, 0.02, 'Mercury.jpg');
createPlanet('Venus', 3, 35, 0xe39e1c, 0.015, 0.015, 'Venus.jpg');
createPlanet('Earth', 3.5, 50, 0x3498db, 0.01, 0.01, 'Earth.jpg');
createPlanet('Mars', 2.5, 65, 0xc0392b, 0.008, 0.008, 'Mars.jpg');


function animate() {
    requestAnimationFrame(animate);
    stats.update();
    orbitControls.update();

    sun.rotation.y += 0.005;
    
    // Update planets
    planets.forEach(planet => {
        // Self rotation
        planet.mesh.rotation.y += planet.props.rotationSpeed;
        
        // Orbit around sun
        planet.group.rotation.y += planet.props.orbitSpeed;
        
        // Update planet distance from sun
        planet.mesh.position.x = planet.props.distance;
    });
    
    // Render the scene with active camera
    renderer.render(scene, activeCamera);
}

animate();






