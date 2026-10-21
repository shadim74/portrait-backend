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
const LEAD = .04;             // the mouth moves a little before the sound is heard

export async function mountAvatar(container, frame, model) {
  const base = model.url.slice(0, model.url.lastIndexOf('/') + 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.domElement.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block';

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
      const part = m.name.includes('head') ? 'head' : 'body';
      const p = model.textures?.[part] || {};
      return new THREE.MeshStandardMaterial({
        name: m.name, roughness: part === 'head' ? 0.62 : 0.85, metalness: 0,
        map: p.color ? tex(p.color) : null,
        normalMap: p.normal ? tex(p.normal, false) : null,
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
  const look = headPos.clone().add(new THREE.Vector3(0, -(model.lookDown ?? 32), 0));
  camera.position.set(look.x, look.y + 8, look.z + (model.distance ?? 275));
  camera.lookAt(look);

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
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize); ro.observe(container);
  container.classList.add('has-model');
  container.appendChild(renderer.domElement);
  resize();

  // --- speech state (set by the page) ---------------------------------------
  let audio = null, timeline = null, tlIdx = 0;
  let talking = false, lastSpeech = -1e9;
  const visW = Object.fromEntries(VISEMES.map(v => [v, 0]));
  let jaw = 0, fbIdx = 0, fbT = 0;

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
      morph(VISEME_MORPH[v], visW[v]);
    }
    jaw = approach(jaw, jawTarget, 20, dt);
    morph('AK_25_JawOpen', jaw);

    // Expression: faint polite smile, eyebrows lift on loud syllables.
    morph('AK_44_MouthSmileLeft', .16); morph('AK_45_MouthSmileRight', .16);
    morph('AK_19_EyeSquintLeft', .2); morph('AK_20_EyeSquintRight', .2);
    const onset = Math.max(0, loud - prevMouth); prevMouth = loud;
    if (speaking && onset > .18 && Math.random() < .35) brow = Math.min(1, brow + .7);
    brow = approach(brow, 0, 2.2, dt);
    morph('AK_03_BrowInnerUp', brow * .55);
    morph('AK_04_BrowOuterUpLeft', brow * .45); morph('AK_05_BrowOuterUpRight', brow * .45);

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
    morph('AK_09_EyeBlinkLeft', lid); morph('AK_10_EyeBlinkRight', lid);

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
    morph('AK_15_EyeLookOutLeft', Math.max(0, gaze.x) * .7); morph('AK_14_EyeLookInRight', Math.max(0, gaze.x) * .7);
    morph('AK_13_EyeLookInLeft', Math.max(0, -gaze.x) * .7); morph('AK_16_EyeLookOutRight', Math.max(0, -gaze.x) * .7);
    morph('AK_11_EyeLookDownLeft', Math.max(0, gaze.y) * .6); morph('AK_12_EyeLookDownRight', Math.max(0, gaze.y) * .6);
    morph('AK_17_EyeLookUpLeft', Math.max(0, -gaze.y) * .6); morph('AK_18_EyeLookUpRight', Math.max(0, -gaze.y) * .6);

    // Head: a springy nod on stressed syllables, and follows the gaze a bit.
    if (speaking && onset > .15) nodVel += onset * 2.2;
    nodVel += (-nod * 60 - nodVel * 9) * dt; nod += nodVel * dt;
    addTurn(head, nod * .12 + gaze.y * .05, gaze.x * .12, 0);
    addTurn(neck, nod * .05, gaze.x * .05, 0);

    renderer.render(scene, camera);
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);

  return {
    /** Start lip-sync for an <audio> element, with the server's mouth-shape timeline. */
    speak(el, visemes) { audio = el; timeline = visemes || null; tlIdx = 0; },
    /** Speech stopped or interrupted. */
    silence() { audio = null; timeline = null; },
    dispose() {
      cancelAnimationFrame(raf); ro.disconnect(); mixer.stopAllAction();
      renderer.dispose(); renderer.domElement.remove(); container.classList.remove('has-model');
      scene.traverse(o => { if (o.isMesh) { o.geometry.dispose(); [].concat(o.material).forEach(m => { m.map?.dispose(); m.normalMap?.dispose(); m.dispose(); }); } });
    },
  };
}
