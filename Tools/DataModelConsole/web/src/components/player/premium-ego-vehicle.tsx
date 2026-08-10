"use client";

import { RoundedBox } from "@react-three/drei";
import { memo, useEffect, useMemo } from "react";
import {
  AdditiveBlending,
  BufferGeometry,
  Color,
  DoubleSide,
  Float32BufferAttribute,
} from "three";

interface BodySection {
  centerY: number;
  halfHeight: number;
  halfWidth: number;
  z: number;
}

const BODY_SECTIONS: readonly BodySection[] = [
  { z: -2.44, halfWidth: 0.12, centerY: 0.53, halfHeight: 0.08 },
  { z: -2.28, halfWidth: 0.76, centerY: 0.56, halfHeight: 0.32 },
  { z: -1.82, halfWidth: 0.98, centerY: 0.58, halfHeight: 0.42 },
  { z: -0.78, halfWidth: 1.06, centerY: 0.59, halfHeight: 0.49 },
  { z: 0.58, halfWidth: 1.07, centerY: 0.57, halfHeight: 0.49 },
  { z: 1.62, halfWidth: 1.02, centerY: 0.54, halfHeight: 0.43 },
  { z: 2.22, halfWidth: 0.84, centerY: 0.51, halfHeight: 0.34 },
  { z: 2.43, halfWidth: 0.1, centerY: 0.49, halfHeight: 0.07 },
];
const HEADLIGHT_COLOR = new Color(3.8, 4.8, 5.2);
const TAILLIGHT_COLOR = new Color(4.5, 0.08, 0.24);
const BODY_SHELL_COLOR = new Color(0.12, 1.45, 2.2);

function signedPower(value: number, exponent: number): number {
  return Math.sign(value) * Math.abs(value) ** exponent;
}

function createBodyGeometry(): BufferGeometry {
  const radialSegments = 40;
  const positions: number[] = [];
  const indices: number[] = [];

  for (const section of BODY_SECTIONS) {
    for (let segment = 0; segment < radialSegments; segment++) {
      const angle = (segment / radialSegments) * Math.PI * 2;
      positions.push(
        section.halfWidth * signedPower(Math.cos(angle), 0.68),
        section.centerY +
          section.halfHeight * signedPower(Math.sin(angle), 0.76),
        section.z,
      );
    }
  }

  for (
    let sectionIndex = 0;
    sectionIndex < BODY_SECTIONS.length - 1;
    sectionIndex++
  ) {
    for (let segment = 0; segment < radialSegments; segment++) {
      const nextSegment = (segment + 1) % radialSegments;
      const current = sectionIndex * radialSegments + segment;
      const currentNext = sectionIndex * radialSegments + nextSegment;
      const forward = (sectionIndex + 1) * radialSegments + segment;
      const forwardNext =
        (sectionIndex + 1) * radialSegments + nextSegment;
      indices.push(
        current,
        currentNext,
        forward,
        currentNext,
        forwardNext,
        forward,
      );
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute(
    "position",
    new Float32BufferAttribute(positions, 3),
  );
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

function PremiumWheel({
  side,
  z,
}: {
  side: -1 | 1;
  z: number;
}) {
  const outerFace = side * 0.17;
  return (
    <group position={[side * 1.02, 0.42, z]}>
      <mesh castShadow receiveShadow rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.4, 0.4, 0.3, 40]} />
        <meshPhysicalMaterial
          clearcoat={0.18}
          color="#030506"
          envMapIntensity={1.7}
          metalness={0.28}
          roughness={0.34}
        />
      </mesh>
      <mesh
        position={[outerFace - side * 0.035, 0, 0]}
        rotation={[0, 0, Math.PI / 2]}
      >
        <cylinderGeometry args={[0.27, 0.27, 0.035, 32]} />
        <meshPhysicalMaterial
          color="#626c72"
          envMapIntensity={2.2}
          metalness={0.92}
          roughness={0.2}
        />
      </mesh>
      <RoundedBox
        args={[0.055, 0.2, 0.1]}
        position={[outerFace, 0.12, 0.18]}
        radius={0.025}
        smoothness={3}
      >
        <meshPhysicalMaterial
          clearcoat={0.8}
          color="#ff2348"
          emissive="#ff153d"
          emissiveIntensity={0.3}
          metalness={0.55}
          roughness={0.2}
        />
      </RoundedBox>
      <group position={[outerFace + side * 0.02, 0, 0]}>
        {Array.from({ length: 6 }, (_, index) => {
          const angle = (index / 6) * Math.PI;
          return (
            <RoundedBox
              key={angle}
              args={[0.045, 0.055, 0.48]}
              rotation={[angle, 0, 0]}
              radius={0.018}
              smoothness={2}
            >
              <meshPhysicalMaterial
                clearcoat={0.9}
                color="#eef3f5"
                envMapIntensity={2.7}
                metalness={0.86}
                roughness={0.12}
              />
            </RoundedBox>
          );
        })}
        <mesh rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.09, 0.09, 0.065, 24]} />
          <meshPhysicalMaterial
            clearcoat={0.75}
            color="#11191d"
            envMapIntensity={2}
            metalness={0.82}
            roughness={0.18}
          />
        </mesh>
      </group>
      <mesh
        position={[outerFace + side * 0.025, 0, 0]}
        rotation={[0, Math.PI / 2, 0]}
      >
        <torusGeometry args={[0.29, 0.032, 12, 40]} />
        <meshPhysicalMaterial
          clearcoat={0.95}
          color="#f8fbfc"
          envMapIntensity={3}
          metalness={0.9}
          roughness={0.1}
        />
      </mesh>
    </group>
  );
}

function SideMirror({ side }: { side: -1 | 1 }) {
  return (
    <group position={[side * 1.08, 1.02, 0.52]}>
      <RoundedBox
        args={[0.28, 0.13, 0.22]}
        castShadow
        radius={0.06}
        smoothness={4}
      >
        <meshPhysicalMaterial
          clearcoat={1}
          clearcoatRoughness={0.04}
          color="#e8edef"
          envMapIntensity={3.2}
          metalness={0.65}
          roughness={0.12}
        />
      </RoundedBox>
      <RoundedBox
        args={[0.06, 0.09, 0.08]}
        position={[-side * 0.11, -0.07, -0.02]}
        radius={0.025}
        smoothness={3}
      >
        <meshPhysicalMaterial
          color="#10191e"
          metalness={0.6}
          roughness={0.2}
        />
      </RoundedBox>
    </group>
  );
}

function GroundSignature() {
  return (
    <>
      <mesh
        position={[0, 0.015, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        scale={[1.52, 2.8, 1]}
      >
        <circleGeometry args={[1, 64]} />
        <meshBasicMaterial
          blending={AdditiveBlending}
          color="#45ddff"
          depthWrite={false}
          opacity={0.08}
          side={DoubleSide}
          toneMapped={false}
          transparent
        />
      </mesh>
      <mesh
        position={[0, 0.025, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        scale={[1.68, 3.05, 1]}
      >
        <ringGeometry args={[0.92, 1, 64]} />
        <meshBasicMaterial
          blending={AdditiveBlending}
          color="#62eaff"
          depthWrite={false}
          opacity={0.5}
          side={DoubleSide}
          toneMapped={false}
          transparent
        />
      </mesh>
    </>
  );
}

export const PremiumEgoVehicle = memo(function PremiumEgoVehicle() {
  const bodyGeometry = useMemo(createBodyGeometry, []);
  useEffect(() => () => bodyGeometry.dispose(), [bodyGeometry]);

  return (
    <group name="auto-e2e-premium-ego-vehicle" position={[0, 0.02, 0]}>
      <GroundSignature />

      <RoundedBox
        args={[1.72, 0.18, 3.86]}
        castShadow
        position={[0, 0.24, 0]}
        radius={0.08}
        receiveShadow
        smoothness={4}
      >
        <meshPhysicalMaterial
          clearcoat={0.45}
          color="#071015"
          envMapIntensity={2}
          metalness={0.78}
          roughness={0.2}
        />
      </RoundedBox>

      <mesh
        castShadow
        geometry={bodyGeometry}
        receiveShadow
      >
        <meshPhysicalMaterial
          clearcoat={1}
          clearcoatRoughness={0.035}
          color="#f6f8f8"
          emissive="#cbd7dc"
          emissiveIntensity={0.09}
          envMapIntensity={1.8}
          iridescence={0.05}
          iridescenceIOR={1.36}
          iridescenceThicknessRange={[120, 420]}
          metalness={0.16}
          roughness={0.15}
          sheen={0.18}
          sheenColor="#def6ff"
          sheenRoughness={0.32}
        />
      </mesh>
      <mesh
        geometry={bodyGeometry}
        scale={[1.025, 1.045, 1.025]}
      >
        <meshBasicMaterial
          blending={AdditiveBlending}
          color={BODY_SHELL_COLOR}
          depthWrite={false}
          opacity={0.022}
          side={DoubleSide}
          toneMapped={false}
          transparent
        />
      </mesh>

      <RoundedBox
        args={[1.46, 0.25, 1.88]}
        position={[0, 0.76, -0.2]}
        radius={0.16}
        smoothness={5}
      >
        <meshPhysicalMaterial
          color="#070c10"
          envMapIntensity={1.6}
          metalness={0.35}
          roughness={0.28}
        />
      </RoundedBox>
      {[-0.43, 0.43].flatMap((x) =>
        [-0.56, 0.38].map((z) => (
          <RoundedBox
            key={`${x}:${z}`}
            args={[0.4, 0.5, 0.42]}
            position={[x, 0.92, z]}
            radius={0.12}
            smoothness={4}
          >
            <meshPhysicalMaterial
              color="#121b21"
              envMapIntensity={1.5}
              metalness={0.24}
              roughness={0.3}
            />
          </RoundedBox>
        )),
      )}

      <mesh
        castShadow
        position={[0, 1.11, -0.22]}
        scale={[0.87, 0.45, 1.42]}
      >
        <sphereGeometry args={[1, 48, 24]} />
        <meshPhysicalMaterial
          attenuationColor="#113244"
          attenuationDistance={0.8}
          clearcoat={1}
          clearcoatRoughness={0.025}
          color="#07131a"
          envMapIntensity={4.2}
          ior={1.48}
          metalness={0.18}
          roughness={0.045}
          thickness={0.42}
          transmission={0.34}
        />
      </mesh>

      {([-1, 1] as const).map((side) => (
        <group key={`ego-side-${side}`}>
          <RoundedBox
            args={[0.045, 0.07, 2.36]}
            position={[side * 0.84, 1.16, -0.2]}
            radius={0.022}
            smoothness={3}
          >
            <meshPhysicalMaterial
              clearcoat={1}
              color="#edf2f3"
              envMapIntensity={3.2}
              metalness={0.7}
              roughness={0.1}
            />
          </RoundedBox>
          <RoundedBox
            args={[0.035, 0.48, 0.1]}
            position={[side * 0.81, 1.13, -0.2]}
            radius={0.018}
            smoothness={3}
          >
            <meshPhysicalMaterial
              color="#101a20"
              metalness={0.58}
              roughness={0.14}
            />
          </RoundedBox>
          <RoundedBox
            args={[0.025, 0.035, 0.42]}
            position={[side * 1.07, 0.81, -0.42]}
            radius={0.012}
            smoothness={2}
          >
            <meshPhysicalMaterial
              color="#17242a"
              envMapIntensity={2}
              metalness={0.82}
              roughness={0.16}
            />
          </RoundedBox>
          <SideMirror side={side} />
          <PremiumWheel side={side} z={-1.48} />
          <PremiumWheel side={side} z={1.48} />
        </group>
      ))}

      {([-0.6, 0.6] as const).map((x) => (
        <RoundedBox
          key={`headlight-${x}`}
          args={[0.58, 0.075, 0.055]}
          position={[x, 0.76, 2.29]}
          rotation={[0, 0, x < 0 ? -0.12 : 0.12]}
          radius={0.03}
          smoothness={4}
        >
          <meshBasicMaterial
            color={HEADLIGHT_COLOR}
            toneMapped={false}
          />
        </RoundedBox>
      ))}
      <RoundedBox
        args={[1.36, 0.08, 0.06]}
        position={[0, 0.7, -2.31]}
        radius={0.035}
        smoothness={4}
      >
        <meshBasicMaterial color={TAILLIGHT_COLOR} toneMapped={false} />
      </RoundedBox>
      <RoundedBox
        args={[1.18, 0.1, 0.055]}
        position={[0, 0.35, 2.35]}
        radius={0.04}
        smoothness={3}
      >
        <meshPhysicalMaterial
          color="#081014"
          envMapIntensity={1.8}
          metalness={0.65}
          roughness={0.24}
        />
      </RoundedBox>

      <mesh
        position={[0, 1.075, 1.42]}
        rotation={[-Math.PI / 2, 0, 0]}
      >
        <ringGeometry args={[0.055, 0.09, 6]} />
        <meshBasicMaterial color="#77f1ff" toneMapped={false} />
      </mesh>
      <mesh position={[0, 0.18, 0]} scale={[0.76, 0.025, 1.95]}>
        <boxGeometry />
        <meshBasicMaterial
          blending={AdditiveBlending}
          color="#46e4ff"
          depthWrite={false}
          opacity={0.28}
          toneMapped={false}
          transparent
        />
      </mesh>
    </group>
  );
});
