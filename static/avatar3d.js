// 3D portrait: renders a rigged model (Microsoft Rocketbox FBX) inside the frame.
//
// - Body and hands: motion-captured Rocketbox clips. An idle loop while
//   listening, talking-with-gestures clips while speaking, a nod afterwards.
// - Lips: the server sends a timeline of mouth shapes (visemes) with each
//   audio clip; we follow it using the audio's playback time. Without a
//   timeline we fall back to the voice loudness (--mouth on the frame).
// - Face: blinking, small eye movements, eyebrow lifts on loud syllables,
//   head nods that follow the speech rhythm.
// Loaded only when a persona has a "model" in personas.json.
import * as THREE from 'three';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

const MORPH_PREFIX = 'blendShape1.';
const VISEME_MORPH = {
  sil: 'AA_VI_00_Sil', PP: 'AA_VI_01_PP', FF: 'AA_VI_02_FF', TH: 'AA_VI_03_TH', DD: 'AA_VI_04_DD',
  kk: 'AA_VI_05_KK', CH: 'AA_VI_06_CH', SS: 'AA_VI_07_SS', nn: 'AA_VI_08_nn', RR: 'AA_VI_09_RR',
  aa: 'AA_VI_10_aa', E: 'AA_VI_11_E', I: 'AA_VI_12_I', O: 'AA_VI_13_O', U: 'AA_VI_14_U',
};
const VISEMES = Object.keys(VISEME_MORPH);
// How much the jaw drops for each mouth shape, on top of the shape itself.
const JAW = { aa: .2, O: .18, E: .15, I: .08, U: .1, RR: .08, CH: .05, kk: .1, DD: .06, nn: .05, TH: .05 };
const FALLBACK_CYCLE = ['aa', 'O', 'E', 'I', 'U', 'DD'];
// Facial expressions (ARKit blend shapes, 0..1), blended smoothly on top of
// everything else. The server tags each spoken piece with one of these.
const EXPRESSIONS = {
  neutral:   { AK_44_MouthSmileLeft: .12, AK_45_MouthSmileRight: .12, AK_19_EyeSquintLeft: .15, AK_20_EyeSquintRight: .15 },
  happy:     { AK_44_MouthSmileLeft: .9, AK_45_MouthSmileRight: .9, AU_12_LipCornerPuller: .55, AU_06_CheekRaiser: .5,
               AK_07_CheekSquintLeft: .4, AK_08_CheekSquintRight: .4, AK_19_EyeSquintLeft: .2, AK_20_EyeSquintRight: .2,
               AU_25_LipsPart: .2 },
  sad:       { AU_01_InnerBrowRaiser: .8, AK_03_BrowInnerUp: .5, AU_15_LipCornerDepressor: .75, AK_30_MouthFrownLeft: .5,
               AK_31_MouthFrownRight: .5, AU_17_ChinRaiser: .35, AU_41_LidDroop: .35,
               AK_11_EyeLookDownLeft: .15, AK_12_EyeLookDownRight: .15 },
  surprised: { AU_01_InnerBrowRaiser: .9, AU_02_OuterBrowRaiser: .9, AK_03_BrowInnerUp: .6, AK_04_BrowOuterUpLeft: .7,
               AK_05_BrowOuterUpRight: .7, AU_05_UpperLidRaiser: .8, AK_21_EyeWideLeft: .6, AK_22_EyeWideRight: .6,
               AU_26_JawDrop: .25, AU_25_LipsPart: .4 },
  angry:     { AU_04_BrowLowerer: .8, AK_01_BrowDownLeft: .5, AK_02_BrowDownRight: .5, AU_07_LidTightener: .25,
               AU_09_NoseWrinkler: .35, AU_23_LipTightener: .5, AU_24_LipPressor: .3, AK_30_MouthFrownLeft: .3,
               AK_31_MouthFrownRight: .3 },
  thinking:  { AU_04_BrowLowerer: .35, AK_05_BrowOuterUpRight: .5, AK_36_MouthPressLeft: .45, AK_33_MouthLeft: .3,
               AU_17_ChinRaiser: .25, AK_17_EyeLookUpLeft: .35, AK_18_EyeLookUpRight: .35,
               AK_15_EyeLookOutLeft: .3, AK_14_EyeLookInRight: .3 },
};
// Head posture for each expression: [nod down, turn, tilt] in radians.
const POSTURE = { neutral: [0, 0, 0], happy: [-.03, 0, .03], sad: [.08, 0, -.02], surprised: [-.05, 0, 0],
                  angry: [.03, 0, 0], thinking: [-.02, .06, .07] };
const EXPRESSION_HOLD_MS = 3500;   // after speaking, keep the last expression this long
const LEAD = .04;             // the mouth moves a little before the sound is heard

export async function mountAvatar(container, frame, model, restingMood = 'neutral') {
  const base = model.url.slice(0, model.url.lastIndexOf('/') + 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.localClippingEnabled = true;
  renderer.domElement.className = 'avatar3d';
  renderer.domElement.style.cssText = 'position:absolute;display:block;pointer-events:none;z-index:2';

  // "Out of frame": the canvas is bigger than the picture and sits on top of
  // the gold frame. Inside the picture everything is drawn; outside it, only
  // what is above the frame's bottom edge (gesturing arms swinging out past
  // the sides) or in front of the picture plane (hands reaching toward you).
  // His lower body stays hidden "behind" the bottom of the frame.
  const POP = model.popOut === false ? { side: 0, top: 0, bottom: 0 }           // extra canvas, in frame sizes
            : { side: .6, top: .25, bottom: .35, ...(model.popOut || {}) };
  const popDepth = model.popOutDepth ?? 19;            // cm in front of the spine (his chest is ~15)
  const keepAboveBottom = new THREE.Plane(), keepInFront = new THREE.Plane(new THREE.Vector3(0, 0, 1), -popDepth);
  const clipPlanes = model.popOut === false ? [] : [keepAboveBottom, keepInFront];

  const scene = new THREE.Scene();
  // Old-master lighting: warm key from the upper left, cool dim fill, rim light.
  scene.add(new THREE.HemisphereLight(0xffe8c8, 0x201810, 0.55));
  const key = new THREE.DirectionalLight(0xffd9a8, 2.6); key.position.set(-60, 220, 140); scene.add(key);
  const fill = new THREE.DirectionalLight(0x9fb2cf, 0.8); fill.position.set(90, 160, 80); scene.add(fill);
  const rim = new THREE.DirectionalLight(0xffe2b0, 1.2); rim.position.set(40, 200, -120); scene.add(rim);
  const camera = new THREE.PerspectiveCamera(22, 4 / 5, 1, 1000);

  // --- model + textures ---------------------------------------------------
  const tex = (name, srgb = true) => {
    const t = new THREE.TextureLoader().load(base + name);
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  };
  const manager = new THREE.LoadingManager();
  manager.setURLModifier(u => (/\.tga$/i.test(u) ? 'data:,' : u)); // textures are set below
  const [obj, clipsJson] = await Promise.all([
    new FBXLoader(manager).loadAsync(model.url),
    model.animations ? fetch(base + model.animations).then(r => r.json()) : null,
  ]);

  const morphMeshes = [];
  obj.traverse(o => {
    if (!o.isMesh) return;
    o.frustumCulled = false;
    const mats = [].concat(o.material).map(m => {
      // Rocketbox materials: <id>_body, <id>_head and, for hair/eyelashes, <id>_opacity.
      const part = m.name.includes('opacity') ? 'opacity' : m.name.includes('head') ? 'head' : 'body';
      const p = model.textures?.[part] || {};
      const hair = part === 'opacity';
      return new THREE.MeshStandardMaterial({
        name: m.name, roughness: part === 'body' ? 0.85 : hair ? 0.7 : 0.62, metalness: 0,
        clippingPlanes: clipPlanes, clipIntersection: true,   // hidden only if outside BOTH planes
        map: p.color ? tex(p.color) : null,
        normalMap: p.normal ? tex(p.normal, false) : null,
        // hair strands: cut out by the texture's alpha (no sorting problems), both sides visible
        ...(hair ? { alphaTest: 0.35, side: THREE.DoubleSide } : {}),
      });
    });
    o.material = mats.length === 1 ? mats[0] : mats;
    if (o.morphTargetDictionary) morphMeshes.push(o);
  });
  scene.add(obj);

  const bone = name => obj.getObjectByName(name);
  const head = bone('Bip01_Head'), neck = bone('Bip01_Neck');

  // --- body animation -------------------------------------------------------
  const mixer = new THREE.AnimationMixer(obj);
  const clips = {};
  for (const [name, json] of Object.entries(clipsJson || {})) {
    const clip = THREE.AnimationClip.parse(json);
    // Keep him standing in one spot: ignore the whole-body root rotation.
    clip.tracks = clip.tracks.filter(t => !/^Bip01(_Footsteps)?\.quaternion$/.test(t.name));
    clips[name] = clip;
  }
  const pick = prefix => Object.keys(clips).filter(n => n.includes(prefix)).map(n => mixer.clipAction(clips[n]));
  // Idle = calm breathing only (the other idle clips shift weight and look around).
  const breathe = pick('breathe');
  const idleActions = breathe.length ? breathe : pick('idle'), talkActions = pick('talk');
  let current = null;
  const fadeTo = (action, seconds = .7, randomStart = true) => {
    if (!action || action === current) return;
    action.reset();
    if (randomStart) action.time = Math.random() * action.getClip().duration;
    action.setEffectiveWeight(1).fadeIn(seconds).play();
    current?.fadeOut(seconds);
    current = action;
  };
  const randomOf = list => list[Math.floor(Math.random() * list.length)];

  if (idleActions.length) {
    fadeTo(randomOf(idleActions), 0);
  } else {
    // No clips: just relax the T-pose arms.
    for (const [name, sign] of [['Bip01_L_UpperArm', 1], ['Bip01_R_UpperArm', -1]]) {
      bone(name)?.rotateOnWorldAxis(new THREE.Vector3(0, 0, 1), sign * -THREE.MathUtils.degToRad(72));
    }
  }
  mixer.update(0);
  obj.updateMatrixWorld(true);

  // --- camera: waist-up portrait so the hands show ---------------------------
  const headPos = new THREE.Vector3(); head.getWorldPosition(headPos);
  const look = headPos.clone().add(new THREE.Vector3(0, -(model.lookDown ?? 27), 0));
  camera.position.set(look.x, look.y + 8, look.z + (model.distance ?? 275));
  camera.lookAt(look);
  camera.updateMatrixWorld(true);
  {
    // The plane through the camera and the picture's bottom edge: things above it are kept.
    camera.updateProjectionMatrix();
    const a = new THREE.Vector3(-1, -1, .5).unproject(camera), b = new THREE.Vector3(1, -1, .5).unproject(camera);
    keepAboveBottom.setFromCoplanarPoints(camera.position, a, b);
    if (keepAboveBottom.distanceToPoint(headPos) < 0) keepAboveBottom.negate();
  }

  // --- helpers ---------------------------------------------------------------
  const morph = (name, value) => {
    for (const m of morphMeshes) {
      const i = m.morphTargetDictionary[MORPH_PREFIX + name];
      if (i !== undefined) m.morphTargetInfluences[i] = value;
    }
  };
  const tmpQ = new THREE.Quaternion(), tmpE = new THREE.Euler();
  const addTurn = (b, x, y, z) => {           // on top of the animated pose
    if (!b) return;
    tmpE.set(x, y, z); b.quaternion.multiply(tmpQ.setFromEuler(tmpE));
  };
  const approach = (cur, target, rate, dt) => cur + (target - cur) * Math.min(1, rate * dt);
  const css = name => parseFloat(getComputedStyle(frame).getPropertyValue(name)) || 0;

  const resize = () => {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    // Canvas = picture area plus margins, positioned over the frame.
    const fr = frame.getBoundingClientRect(), cr = container.getBoundingClientRect();
    const left = Math.round(w * POP.side), top = Math.round(h * POP.top), bottom = Math.round(h * POP.bottom);
    const W = w + 2 * left, H = h + top + bottom;
    Object.assign(renderer.domElement.style, {
      left: `${cr.left - fr.left - left}px`, top: `${cr.top - fr.top - top}px`, width: `${W}px`, height: `${H}px`,
    });
    renderer.setSize(W, H, false);
    // Same framing as before inside the picture, extended outward.
    camera.aspect = w / h;
    camera.setViewOffset(w, h, -left, -top, W, H);
  };
  const ro = new ResizeObserver(resize); ro.observe(container);
  container.classList.add('has-model');
  frame.appendChild(renderer.domElement);
  resize();

  // --- speech state (set by the page) ---------------------------------------
  let audio = null, timeline = null, tlIdx = 0;
  let talking = false, lastSpeech = -1e9;
  const visW = Object.fromEntries(VISEMES.map(v => [v, 0]));
  let jaw = 0, fbIdx = 0, fbT = 0;

  // expression state: one weight per expression, eased toward the current one
  let resting = EXPRESSIONS[restingMood] ? restingMood : 'neutral';
  let emotion = resting, lastExpressed = 0;
  const exprW = Object.fromEntries(Object.keys(EXPRESSIONS).map(e => [e, e === emotion ? 1 : 0]));

  // All face drivers add into this per frame; it is written to the mesh once.
  const face = new Map(), touched = new Set();
  const add = (name, v) => { face.set(name, (face.get(name) || 0) + v); touched.add(name); };

  // face state
  let nextBlink = performance.now() + 1500, blinkStart = -1, doubleBlink = false;
  let gaze = { x: 0, y: 0 }, gazeTarget = { x: 0, y: 0 }, nextGaze = 0;
  let brow = 0, nod = 0, nodVel = 0, prevMouth = 0;

  const clock = new THREE.Clock();
  let raf = 0;
  const loop = () => {
    const dt = Math.min(.05, clock.getDelta());
    const now = performance.now();
    const loud = css('--mouth');                    // 0..1 voice loudness
    const idle = frame.classList.contains('idle');
    const speaking = !!audio && !audio.paused && !audio.ended;
    if (speaking) lastSpeech = now;

    // Body: talking gestures while speaking (a short grace period bridges the
    // gap between sentences), then settle straight back into calm breathing.
    const shouldTalk = now - lastSpeech < 400;
    if (shouldTalk !== talking) {
      talking = shouldTalk;
      if (talking && talkActions.length) fadeTo(randomOf(talkActions), .5);
      else if (!talking && idleActions.length) fadeTo(randomOf(idleActions), .6);
    }
    mixer.update(dt);

    // Lips -----------------------------------------------------------------
    const targets = Object.fromEntries(VISEMES.map(v => [v, 0]));
    let jawTarget = 0;
    if (speaking && timeline?.length) {
      const t = audio.currentTime + LEAD;
      if (t < (timeline[tlIdx]?.[0] ?? 0)) tlIdx = 0;
      while (tlIdx + 1 < timeline.length && timeline[tlIdx + 1][0] <= t) tlIdx++;
      const vis = t >= timeline[0][0] ? timeline[tlIdx][1] : 'sil';
      const strength = .65 + .5 * Math.min(1, loud * 1.4);   // louder = bigger shapes
      if (vis !== 'sil') targets[vis] = Math.min(1, strength);
      jawTarget = (JAW[vis] || 0) * strength;
    } else if (speaking || loud > .05) {
      // no timeline: cycle through shapes in time with the loudness
      fbT += dt;
      if (loud > .08 && fbT > .11) { fbT = 0; fbIdx = (fbIdx + 1 + Math.floor(Math.random() * 3)) % FALLBACK_CYCLE.length; }
      targets[FALLBACK_CYCLE[fbIdx]] = Math.min(1, loud * 1.25);
      jawTarget = loud * .3;
    }
    for (const v of VISEMES) {
      // open quickly, close a bit slower: looks like real articulation
      visW[v] = approach(visW[v], targets[v], targets[v] > visW[v] ? 30 : 18, dt);
      add(VISEME_MORPH[v], visW[v]);
    }
    jaw = approach(jaw, jawTarget, 20, dt);
    add('AK_25_JawOpen', jaw);

    // Expression: ease toward the current one; drift back to the resting mood
    // a few seconds after he stops talking.
    if (!talking && emotion !== resting && now - lastExpressed > EXPRESSION_HOLD_MS) emotion = resting;
    const posture = [0, 0, 0];
    for (const [e, preset] of Object.entries(EXPRESSIONS)) {
      exprW[e] = approach(exprW[e], e === emotion ? 1 : 0, 4, dt);
      if (exprW[e] < .002) continue;
      for (const [name, v] of Object.entries(preset)) add(name, v * exprW[e]);
      POSTURE[e].forEach((v, i) => { posture[i] += v * exprW[e]; });
    }
    // Eyebrows lift on loud syllables.
    const onset = Math.max(0, loud - prevMouth); prevMouth = loud;
    if (speaking && onset > .18 && Math.random() < .35) brow = Math.min(1, brow + .7);
    brow = approach(brow, 0, 2.2, dt);
    add('AK_03_BrowInnerUp', brow * .55);
    add('AK_04_BrowOuterUpLeft', brow * .45); add('AK_05_BrowOuterUpRight', brow * .45);

    // Blinks: every 2-6 s, sometimes double, and more often while talking.
    if (now > nextBlink && blinkStart < 0) blinkStart = now;
    let lid = idle ? .55 : .1;                       // half-asleep when disconnected
    if (blinkStart >= 0) {
      const t = (now - blinkStart) / 170;
      lid = Math.max(lid, t < 1 ? Math.sin(t * Math.PI) : 0);
      if (t >= 1) {
        blinkStart = -1;
        if (!doubleBlink && Math.random() < .2) { doubleBlink = true; nextBlink = now + 90; }
        else { doubleBlink = false; nextBlink = now + (talking ? 1500 : 2200) + Math.random() * 3800; }
      }
    }
    add('AK_09_EyeBlinkLeft', lid); add('AK_10_EyeBlinkRight', lid);

    // Eyes: mostly on the visitor, with small glances; looks aside to "think".
    if (now > nextGaze) {
      // Glances aside only while talking (as if thinking); when listening he
      // keeps his eyes on the visitor with tiny natural movements.
      const away = talking && Math.random() < .25;
      gazeTarget = away ? { x: (Math.random() - .5) * 1.0, y: (Math.random() - .3) * .5 }
                        : { x: (Math.random() - .5) * .2, y: (Math.random() - .5) * .15 };
      nextGaze = now + (away ? 600 + Math.random() * 900 : 1200 + Math.random() * 2500);
    }
    gaze.x = approach(gaze.x, gazeTarget.x, 22, dt); gaze.y = approach(gaze.y, gazeTarget.y, 22, dt);
    add('AK_15_EyeLookOutLeft', Math.max(0, gaze.x) * .7); add('AK_14_EyeLookInRight', Math.max(0, gaze.x) * .7);
    add('AK_13_EyeLookInLeft', Math.max(0, -gaze.x) * .7); add('AK_16_EyeLookOutRight', Math.max(0, -gaze.x) * .7);
    add('AK_11_EyeLookDownLeft', Math.max(0, gaze.y) * .6); add('AK_12_EyeLookDownRight', Math.max(0, gaze.y) * .6);
    add('AK_17_EyeLookUpLeft', Math.max(0, -gaze.y) * .6); add('AK_18_EyeLookUpRight', Math.max(0, -gaze.y) * .6);

    // Head: a springy nod on stressed syllables, and follows the gaze a bit.
    if (speaking && onset > .15) nodVel += onset * 2.2;
    nodVel += (-nod * 60 - nodVel * 9) * dt; nod += nodVel * dt;
    addTurn(head, nod * .12 + gaze.y * .05 + posture[0], gaze.x * .12 + posture[1], posture[2]);
    addTurn(neck, nod * .05, gaze.x * .05, 0);

    for (const name of touched) morph(name, Math.min(1, face.get(name) || 0));
    face.clear();

    renderer.render(scene, camera);
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);

  return {
    /** Start lip-sync for an <audio> element, with the server's mouth-shape timeline. */
    speak(el, visemes) { audio = el; timeline = visemes || null; tlIdx = 0; },
    /** Speech stopped or interrupted. */
    silence() { audio = null; timeline = null; },
    /** Show a facial expression: neutral, happy, sad, surprised, angry, thinking. */
    express(e) { if (EXPRESSIONS[e]) { emotion = e; lastExpressed = performance.now(); } },
    /** The expression he goes back to when quiet. */
    setRestingMood(e) { if (EXPRESSIONS[e]) resting = e; },
    dispose() {
      cancelAnimationFrame(raf); ro.disconnect(); mixer.stopAllAction();
      renderer.dispose(); renderer.domElement.remove(); container.classList.remove('has-model');
      scene.traverse(o => { if (o.isMesh) { o.geometry.dispose(); [].concat(o.material).forEach(m => { m.map?.dispose(); m.normalMap?.dispose(); m.dispose(); }); } });
    },
  };
}
