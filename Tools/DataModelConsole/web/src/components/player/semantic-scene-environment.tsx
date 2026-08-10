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
const SEMANTIC_GLOW = new THREE.Color(1.45, 1.45, 1.45);

interface GroundTextures {
  asphalt: THREE.DataTexture;
  roughness: THREE.DataTexture;
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
  const roughnessPixels = new Uint8Array(size * size * 4);

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

      const roughness = Math.max(166, 245 - grain * 2 - aggregate);
      roughnessPixels[offset] = roughness;
      roughnessPixels[offset + 1] = roughness;
      roughnessPixels[offset + 2] = roughness;
      roughnessPixels[offset + 3] = 255;
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
  configureGroundTexture(asphalt);
  configureGroundTexture(roughness);
  roughness.colorSpace = THREE.NoColorSpace;
  return { asphalt, roughness };
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
      edgePixels[pixelOffset + 3] = Math.min(
        255,
        96 + pixels[pixelOffset + 3],
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
      groundTextures.roughness.dispose();
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
          blur={compact ? [72, 20] : [144, 36]}
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
          resolution={compact ? 128 : 256}
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
          displacementScale={0.16}
          emissive="#ffffff"
          emissiveIntensity={0.035}
          emissiveMap={semanticTextures.raster}
          envMapIntensity={0.42}
          map={semanticTextures.raster}
          metalness={0.18}
          opacity={0.34}
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

export function SceneEnvironment({
  groundCenterZ,
  groundLength,
  groundWidth,
}: {
  groundCenterZ: number;
  groundLength: number;
  groundWidth: number;
}) {
  const { size } = useThree();
  const compact = size.width < 700;

  return (
    <>
      <color attach="background" args={["#050d13"]} />
      <fog attach="fog" args={["#07131a", 92, 236]} />
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
            colors={["#02050b", "#061522", "#173344", "#744b5d"]}
            size={1024}
            stops={[0, 0.38, 0.7, 1]}
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
        shadow-mapSize-height={compact ? 512 : 1024}
        shadow-mapSize-width={compact ? 512 : 1024}
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
      <ContactShadows
        blur={2.6}
        color="#020609"
        far={8}
        frames={1}
        height={groundLength}
        opacity={0.48}
        position={[0, 0.045, groundCenterZ]}
        resolution={compact ? 256 : 512}
        smooth
        width={groundWidth}
      />
    </>
  );
}

export function ScenePostProcessing() {
  const { camera, scene, size } = useThree();
  const compact = size.width < 700;
  const passes = useMemo(() => {
    const ssao = new SSAOPass(scene, camera, size.width, size.height);
    ssao.kernelRadius = compact ? 5 : 8;
    ssao.minDistance = 0.0015;
    ssao.maxDistance = 0.085;

    const bloom = new UnrealBloomPass(
      new THREE.Vector2(size.width, size.height),
      compact ? 0.17 : 0.24,
      0.52,
      0.88,
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
    toneMapping.uniforms.exposure.value = 0.66;
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
  }, [camera, compact, scene, size.height, size.width]);

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
