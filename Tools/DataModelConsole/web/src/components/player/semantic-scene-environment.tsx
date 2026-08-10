"use client";

import {
  ContactShadows,
  Effects,
  Environment,
  GradientTexture,
  Grid,
  MeshReflectorMaterial,
} from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo } from "react";
import * as THREE from "three";
import {
  ACESFilmicToneMappingShader,
  BrightnessContrastShader,
  GammaCorrectionShader,
  HueSaturationShader,
  ShaderPass,
  SSAOPass,
  UnrealBloomPass,
  VignetteShader,
} from "three-stdlib";

import type { SemanticOccupancyArtifact } from "@/lib/semantic-occupancy";

const HDRI_PATH =
  "/assets/semantic-occupancy/poly-haven/studio_small_09_1k.hdr";
const ASPHALT_NORMAL_SCALE = new THREE.Vector2(0.24, 0.24);
const SEMANTIC_GLOW = new THREE.Color(1.45, 1.45, 1.45);

interface GroundTextures {
  asphalt: THREE.DataTexture;
  normal: THREE.DataTexture;
  roughness: THREE.DataTexture;
  wetness: THREE.DataTexture;
}

interface SemanticTextures {
  confidence: THREE.DataTexture;
  edge: THREE.DataTexture;
  raster: THREE.DataTexture;
}

function configureGroundTexture(texture: THREE.DataTexture) {
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(30, 45);
  texture.needsUpdate = true;
}

function createGroundTextures(): GroundTextures {
  const size = 192;
  const asphaltPixels = new Uint8Array(size * size * 4);
  const normalPixels = new Uint8Array(size * size * 4);
  const roughnessPixels = new Uint8Array(size * size * 4);
  const wetnessPixels = new Uint8Array(size * size * 4);

  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const offset = (row * size + col) * 4;
      const hash =
        Math.imul(row + 31, 73856093) ^
        Math.imul(col + 17, 19349663) ^
        Math.imul(row + col + 11, 83492791);
      const grain = Math.abs(hash % 24);
      const aggregate = hash % 113 === 0 ? 28 : 0;
      const base = 31 + grain + aggregate;
      asphaltPixels[offset] = Math.min(91, base - 3);
      asphaltPixels[offset + 1] = Math.min(96, base + 1);
      asphaltPixels[offset + 2] = Math.min(101, base + 3);
      asphaltPixels[offset + 3] = 255;

      const normalX = ((hash >>> 5) & 31) - 15;
      const normalY = ((hash >>> 13) & 31) - 15;
      normalPixels[offset] = 128 + normalX;
      normalPixels[offset + 1] = 128 + normalY;
      normalPixels[offset + 2] = 244;
      normalPixels[offset + 3] = 255;

      const roughness = Math.max(166, 245 - grain * 2 - aggregate);
      roughnessPixels[offset] = roughness;
      roughnessPixels[offset + 1] = roughness;
      roughnessPixels[offset + 2] = roughness;
      roughnessPixels[offset + 3] = 255;

      const broadPatch =
        Math.sin(row * 0.105) +
        Math.cos(col * 0.083) +
        Math.sin((row + col) * 0.047);
      const wetness =
        broadPatch > 1.18 && Math.abs(hash % 17) > 2 ? 210 : 0;
      wetnessPixels[offset] = wetness;
      wetnessPixels[offset + 1] = wetness;
      wetnessPixels[offset + 2] = wetness;
      wetnessPixels[offset + 3] = 255;
    }
  }

  const asphalt = new THREE.DataTexture(
    asphaltPixels,
    size,
    size,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  const roughness = new THREE.DataTexture(
    roughnessPixels,
    size,
    size,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  const normal = new THREE.DataTexture(
    normalPixels,
    size,
    size,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  const wetness = new THREE.DataTexture(
    wetnessPixels,
    size,
    size,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  configureGroundTexture(asphalt);
  configureGroundTexture(normal);
  configureGroundTexture(roughness);
  configureGroundTexture(wetness);
  normal.colorSpace = THREE.NoColorSpace;
  roughness.colorSpace = THREE.NoColorSpace;
  wetness.colorSpace = THREE.NoColorSpace;
  return { asphalt, normal, roughness, wetness };
}

function configureSemanticTexture(
  texture: THREE.DataTexture,
  colorSpace: THREE.ColorSpace,
) {
  texture.colorSpace = colorSpace;
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.wrapS = THREE.RepeatWrapping;
  texture.repeat.x = -1;
  texture.offset.x = 1;
  texture.needsUpdate = true;
}

function isSemanticEdge(
  pixels: Uint8Array,
  width: number,
  height: number,
  row: number,
  col: number,
): boolean {
  const offset = (row * width + col) * 4;
  if (pixels[offset + 3] === 0) return false;
  const neighbors = [
    [row - 1, col],
    [row + 1, col],
    [row, col - 1],
    [row, col + 1],
  ] as const;
  return neighbors.some(([neighborRow, neighborCol]) => {
    if (
      neighborRow < 0 ||
      neighborRow >= height ||
      neighborCol < 0 ||
      neighborCol >= width
    ) {
      return true;
    }
    const neighborOffset = (neighborRow * width + neighborCol) * 4;
    if (pixels[neighborOffset + 3] === 0) return true;
    return (
      Math.abs(pixels[offset] - pixels[neighborOffset]) +
        Math.abs(pixels[offset + 1] - pixels[neighborOffset + 1]) +
        Math.abs(pixels[offset + 2] - pixels[neighborOffset + 2]) >
      72
    );
  });
}

function createSemanticTextures(
  artifact: SemanticOccupancyArtifact,
  pixels: Uint8Array,
): SemanticTextures {
  const confidencePixels = new Uint8Array(
    artifact.width * artifact.height,
  );
  const edgePixels = new Uint8Array(
    artifact.width * artifact.height * 4,
  );

  for (let row = 0; row < artifact.height; row++) {
    for (let col = 0; col < artifact.width; col++) {
      const pixelOffset = (row * artifact.width + col) * 4;
      confidencePixels[row * artifact.width + col] =
        pixels[pixelOffset + 3];
      if (
        !isSemanticEdge(
          pixels,
          artifact.width,
          artifact.height,
          row,
          col,
        )
      ) {
        continue;
      }
      edgePixels[pixelOffset] = pixels[pixelOffset];
      edgePixels[pixelOffset + 1] = pixels[pixelOffset + 1];
      edgePixels[pixelOffset + 2] = pixels[pixelOffset + 2];
      const red = pixels[pixelOffset];
      const green = pixels[pixelOffset + 1];
      const blue = pixels[pixelOffset + 2];
      const importantClass =
        (green > red * 1.45 && green > blue * 1.18) ||
        (red > 180 && blue > 120) ||
        (red > 180 && green < 170);
      edgePixels[pixelOffset + 3] = Math.round(
        pixels[pixelOffset + 3] * (importantClass ? 0.92 : 0.42),
      );
    }
  }

  const raster = new THREE.DataTexture(
    pixels,
    artifact.width,
    artifact.height,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  const edge = new THREE.DataTexture(
    edgePixels,
    artifact.width,
    artifact.height,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  const confidence = new THREE.DataTexture(
    confidencePixels,
    artifact.width,
    artifact.height,
    THREE.RedFormat,
    THREE.UnsignedByteType,
  );
  configureSemanticTexture(raster, THREE.SRGBColorSpace);
  configureSemanticTexture(edge, THREE.SRGBColorSpace);
  configureSemanticTexture(confidence, THREE.NoColorSpace);
  return { confidence, edge, raster };
}

export function SemanticGround({
  artifact,
  metersPerCell,
  pixels,
}: {
  artifact: SemanticOccupancyArtifact;
  metersPerCell: number;
  pixels: Uint8Array;
}) {
  const { size } = useThree();
  const groundTextures = useMemo(createGroundTextures, []);
  const semanticTextures = useMemo(
    () => createSemanticTextures(artifact, pixels),
    [artifact, pixels],
  );

  useEffect(
    () => () => {
      groundTextures.asphalt.dispose();
      groundTextures.normal.dispose();
      groundTextures.roughness.dispose();
      groundTextures.wetness.dispose();
    },
    [groundTextures],
  );
  useEffect(
    () => () => {
      semanticTextures.confidence.dispose();
      semanticTextures.edge.dispose();
      semanticTextures.raster.dispose();
    },
    [semanticTextures],
  );

  const groundWidth = artifact.width * metersPerCell;
  const groundLength = artifact.height * metersPerCell;
  const groundCenterZ =
    (artifact.height * (2 / 3) - artifact.height / 2) *
    metersPerCell;
  const widthSegments = Math.min(artifact.width - 1, 180);
  const lengthSegments = Math.min(artifact.height - 1, 220);
  const compact = size.width < 700;

  return (
    <>
      <mesh
        receiveShadow
        position={[0, -0.1, groundCenterZ]}
        rotation={[-Math.PI / 2, 0, 0]}
      >
        <planeGeometry args={[groundWidth, groundLength]} />
        <meshPhysicalMaterial
          clearcoat={0.16}
          clearcoatRoughness={0.54}
          color="#536066"
          envMapIntensity={0.65}
          map={groundTextures.asphalt}
          metalness={0.14}
          normalMap={groundTextures.normal}
          normalScale={ASPHALT_NORMAL_SCALE}
          roughness={0.9}
          roughnessMap={groundTextures.roughness}
        />
      </mesh>

      <mesh
        position={[0, -0.07, groundCenterZ]}
        rotation={[-Math.PI / 2, 0, 0]}
      >
        <planeGeometry args={[groundWidth, groundLength]} />
        <MeshReflectorMaterial
          alphaMap={groundTextures.wetness}
          blur={compact ? [64, 18] : [180, 42]}
          color="#203038"
          depthScale={0.28}
          depthWrite={false}
          maxDepthThreshold={1.2}
          metalness={0.5}
          minDepthThreshold={0.22}
          mirror={0.32}
          mixBlur={0.82}
          mixStrength={1.45}
          opacity={0.14}
          resolution={compact ? 128 : 512}
          roughness={0.48}
          transparent
        />
      </mesh>

      <Grid
        args={[groundWidth, groundLength]}
        cellColor="#23424b"
        cellSize={2}
        cellThickness={0.25}
        fadeDistance={190}
        fadeStrength={1.35}
        followCamera={false}
        infiniteGrid={false}
        position={[0, -0.035, groundCenterZ]}
        sectionColor="#4a7782"
        sectionSize={10}
        sectionThickness={0.58}
      />

      <mesh
        position={[0, 0, groundCenterZ]}
        rotation={[-Math.PI / 2, 0, 0]}
      >
        <planeGeometry
          args={[
            groundWidth,
            groundLength,
            widthSegments,
            lengthSegments,
          ]}
        />
        <meshPhysicalMaterial
          alphaTest={0.01}
          clearcoat={0.35}
          clearcoatRoughness={0.28}
          depthWrite={false}
          displacementMap={semanticTextures.confidence}
          displacementScale={0.12}
          emissive="#ffffff"
          emissiveIntensity={0.008}
          emissiveMap={semanticTextures.raster}
          envMapIntensity={0.42}
          map={semanticTextures.raster}
          metalness={0.18}
          opacity={0.28}
          roughness={0.58}
          side={THREE.DoubleSide}
          transparent
        />
      </mesh>

      <mesh
        position={[0, 0.045, groundCenterZ]}
        rotation={[-Math.PI / 2, 0, 0]}
      >
        <planeGeometry args={[groundWidth, groundLength]} />
        <meshBasicMaterial
          alphaTest={0.01}
          blending={THREE.AdditiveBlending}
          color={SEMANTIC_GLOW}
          depthWrite={false}
          map={semanticTextures.edge}
          side={THREE.DoubleSide}
          toneMapped={false}
          transparent
        />
      </mesh>
    </>
  );
}

const DEMO_STREETLIGHT_Z = [-42, -14, 14, 42, 70, 98] as const;

function DemoStreetlight({
  side,
  z,
}: {
  side: -1 | 1;
  z: number;
}) {
  return (
    <group position={[side * 43.5, 0, z]}>
      <mesh castShadow position={[0, 3.2, 0]}>
        <cylinderGeometry args={[0.08, 0.13, 6.4, 12]} />
        <meshPhysicalMaterial
          color="#647078"
          envMapIntensity={1.8}
          metalness={0.86}
          roughness={0.25}
        />
      </mesh>
      <mesh position={[-side * 1.05, 6.18, 0]}>
        <boxGeometry args={[2.1, 0.1, 0.12]} />
        <meshPhysicalMaterial
          color="#59656c"
          metalness={0.82}
          roughness={0.28}
        />
      </mesh>
      <mesh position={[-side * 2.02, 6.02, 0]}>
        <boxGeometry args={[0.36, 0.14, 0.28]} />
        <meshBasicMaterial
          color={new THREE.Color(3.8, 3.25, 2.2)}
          toneMapped={false}
        />
      </mesh>
      {Math.abs(z - 14) <= 28 && (
        <pointLight
          color="#ffe9bd"
          decay={2}
          distance={18}
          intensity={24}
          position={[-side * 2.02, 5.8, 0]}
        />
      )}
    </group>
  );
}

function DemoTrafficLight({
  side,
}: {
  side: -1 | 1;
}) {
  return (
    <group position={[side * 37.5, 0, 30]} rotation={[0, -side * Math.PI / 2, 0]}>
      <mesh castShadow position={[0, 3.4, 0]}>
        <cylinderGeometry args={[0.1, 0.16, 6.8, 12]} />
        <meshPhysicalMaterial
          color="#334047"
          metalness={0.84}
          roughness={0.24}
        />
      </mesh>
      <mesh position={[0, 6.45, 2.8]}>
        <boxGeometry args={[0.13, 0.13, 5.6]} />
        <meshPhysicalMaterial
          color="#334047"
          metalness={0.84}
          roughness={0.24}
        />
      </mesh>
      <group position={[0, 5.85, 5.38]}>
        <mesh castShadow>
          <boxGeometry args={[0.44, 1.45, 0.42]} />
          <meshPhysicalMaterial
            color="#11191d"
            metalness={0.58}
            roughness={0.3}
          />
        </mesh>
        {[
          { color: new THREE.Color(4.5, 0.04, 0.08), y: 0.45 },
          { color: new THREE.Color(0.25, 0.12, 0.02), y: 0 },
          { color: new THREE.Color(0.02, 0.22, 0.08), y: -0.45 },
        ].map((lamp) => (
          <mesh
            key={lamp.y}
            position={[0, lamp.y, 0.221]}
          >
            <circleGeometry args={[0.13, 20]} />
            <meshBasicMaterial color={lamp.color} toneMapped={false} />
          </mesh>
        ))}
      </group>
    </group>
  );
}

function DemoStreetscape({
  groundCenterZ,
  groundLength,
}: {
  groundCenterZ: number;
  groundLength: number;
}) {
  const railPosts = useMemo(
    () =>
      Array.from(
        { length: Math.floor(groundLength / 8) },
        (_, index) => groundCenterZ - groundLength / 2 + index * 8 + 4,
      ),
    [groundCenterZ, groundLength],
  );

  return (
    <group name="mock-only-streetscape">
      {([-1, 1] as const).map((side) => (
        <group key={`demo-road-edge-${side}`}>
          <mesh
            receiveShadow
            position={[side * 41.2, 0.02, groundCenterZ]}
          >
            <boxGeometry args={[1.2, 0.24, groundLength]} />
            <meshPhysicalMaterial
              color="#899196"
              envMapIntensity={0.75}
              metalness={0.08}
              roughness={0.82}
            />
          </mesh>
          <mesh
            receiveShadow
            position={[side * 44.1, 0.04, groundCenterZ]}
          >
            <boxGeometry args={[4.6, 0.16, groundLength]} />
            <meshPhysicalMaterial
              color="#404b50"
              envMapIntensity={0.55}
              metalness={0.1}
              roughness={0.9}
            />
          </mesh>
          <mesh
            castShadow
            position={[side * 42.25, 0.7, groundCenterZ]}
          >
            <boxGeometry args={[0.12, 0.18, groundLength]} />
            <meshPhysicalMaterial
              color="#9aa6ab"
              envMapIntensity={1.9}
              metalness={0.9}
              roughness={0.2}
            />
          </mesh>
          {railPosts.map((z) => (
            <mesh
              key={`guardrail-${side}-${z}`}
              castShadow
              position={[side * 42.25, 0.4, z]}
            >
              <boxGeometry args={[0.14, 0.8, 0.14]} />
              <meshPhysicalMaterial
                color="#707d83"
                metalness={0.84}
                roughness={0.26}
              />
            </mesh>
          ))}
          {DEMO_STREETLIGHT_Z.map((z) => (
            <DemoStreetlight key={`streetlight-${side}-${z}`} side={side} z={z} />
          ))}
          <DemoTrafficLight side={side} />
        </group>
      ))}
    </group>
  );
}

export function SceneEnvironment({
  groundCenterZ,
  groundLength,
  groundWidth,
  showDemoStreetscape = false,
  viewMode = "orbit",
}: {
  groundCenterZ: number;
  groundLength: number;
  groundWidth: number;
  showDemoStreetscape?: boolean;
  viewMode?: "ego" | "orbit" | "top";
}) {
  const { gl, invalidate, size } = useThree();
  const compact = size.width < 700;
  const fogNear =
    viewMode === "ego" ? 106 : viewMode === "top" ? 132 : 92;
  const fogFar =
    viewMode === "ego" ? 250 : viewMode === "top" ? 290 : 236;

  useEffect(() => {
    gl.toneMappingExposure =
      viewMode === "ego" ? 1.13 : viewMode === "top" ? 1.02 : 1.08;
    invalidate();
  }, [gl, invalidate, viewMode]);

  return (
    <>
      <color attach="background" args={["#050d13"]} />
      <fog attach="fog" args={["#07131a", fogNear, fogFar]} />
      <mesh
        frustumCulled={false}
        position={[0, 28, 45]}
        renderOrder={-100}
      >
        <sphereGeometry args={[350, 32, 18]} />
        <meshBasicMaterial
          depthWrite={false}
          fog={false}
          side={THREE.BackSide}
          toneMapped={false}
        >
          <GradientTexture
            colors={[
              "#02050a",
              "#07131c",
              "#6b4d61",
              "#071522",
              "#02050b",
            ]}
            size={1024}
            stops={[0, 0.36, 0.5, 0.64, 1]}
          />
        </meshBasicMaterial>
      </mesh>
      <Suspense fallback={null}>
        <Environment
          background={false}
          environmentIntensity={0.72}
          environmentRotation={[0, 0.42, 0]}
          files={HDRI_PATH}
        />
      </Suspense>
      <ambientLight intensity={0.1} />
      <hemisphereLight args={["#a5d7e5", "#030607", 0.42]} />
      <directionalLight
        castShadow
        color="#f5fbff"
        intensity={1.05}
        position={[-22, 42, -14]}
        shadow-bias={-0.00018}
        shadow-camera-bottom={-85}
        shadow-camera-far={190}
        shadow-camera-left={-90}
        shadow-camera-near={1}
        shadow-camera-right={90}
        shadow-camera-top={145}
        shadow-mapSize-height={compact ? 384 : 1536}
        shadow-mapSize-width={compact ? 384 : 1536}
      />
      <directionalLight
        color="#62e9ff"
        intensity={0.58}
        position={[30, 18, -12]}
      />
      <directionalLight
        color="#ff476f"
        intensity={0.36}
        position={[-26, 9, 25]}
      />
      {showDemoStreetscape && (
        <DemoStreetscape
          groundCenterZ={groundCenterZ}
          groundLength={groundLength}
        />
      )}
      <ContactShadows
        blur={2.6}
        color="#020609"
        far={8}
        frames={1}
        height={groundLength}
        opacity={0.48}
        position={[0, 0.045, groundCenterZ]}
        resolution={compact ? 192 : 768}
        smooth
        width={groundWidth}
      />
    </>
  );
}

export function ScenePostProcessing({
  viewMode = "orbit",
}: {
  viewMode?: "ego" | "orbit" | "top";
}) {
  const { camera, scene, size } = useThree();
  const compact = size.width < 700;
  const passes = useMemo(() => {
    const ssao = new SSAOPass(scene, camera, size.width, size.height);
    ssao.kernelRadius = compact ? 5 : 8;
    ssao.minDistance = 0.0015;
    ssao.maxDistance = 0.085;

    const bloom = new UnrealBloomPass(
      new THREE.Vector2(size.width, size.height),
      compact
        ? 0.14
        : viewMode === "ego"
          ? 0.28
          : viewMode === "top"
            ? 0.16
            : 0.22,
      viewMode === "ego" ? 0.58 : 0.48,
      viewMode === "top" ? 0.96 : 0.9,
    );

    const contrast = new ShaderPass(BrightnessContrastShader);
    contrast.uniforms.brightness.value = -0.008;
    contrast.uniforms.contrast.value = 0.045;

    const saturation = new ShaderPass(HueSaturationShader);
    saturation.uniforms.saturation.value = 0.08;

    const vignette = new ShaderPass(VignetteShader);
    vignette.uniforms.offset.value = 1.02;
    vignette.uniforms.darkness.value = 0.78;

    const toneMapping = new ShaderPass(ACESFilmicToneMappingShader);
    toneMapping.uniforms.exposure.value =
      viewMode === "ego" ? 0.7 : viewMode === "top" ? 0.62 : 0.66;
    const gamma = new ShaderPass(GammaCorrectionShader);

    return {
      bloom,
      contrast,
      gamma,
      saturation,
      ssao,
      toneMapping,
      vignette,
    };
  }, [camera, compact, scene, size.height, size.width, viewMode]);

  useEffect(
    () => () => {
      Object.values(passes).forEach((pass) => pass.dispose());
    },
    [passes],
  );

  return (
    <Effects
      disableGamma
      multisamping={compact ? 0 : 4}
    >
      <primitive object={passes.ssao} />
      <primitive object={passes.bloom} />
      <primitive object={passes.contrast} />
      <primitive object={passes.saturation} />
      <primitive object={passes.vignette} />
      <primitive object={passes.toneMapping} />
      <primitive object={passes.gamma} />
    </Effects>
  );
}
