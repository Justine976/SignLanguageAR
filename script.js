/* SignAR FSL decoder foundation. The decoder never invents model predictions.
   Model input contract: 30 frames × 456 normalized features; output classes in model/vocabulary.json. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const video=$('video'), handCanvas=$('handCanvas'), handCtx=handCanvas.getContext('2d');
  const poseCanvas=$('poseCanvas'), poseCtx=poseCanvas?.getContext('2d');
  const processCanvas=document.createElement('canvas'), processCtx=processCanvas.getContext('2d');
  const cameraStage=$('cameraStage'), placeholder=$('cameraPlaceholder'), status=$('cameraStatus');
  const liveBadge=$('liveBadge'), trackingBadge=$('trackingBadge'), guide=$('positionGuide');
  const startBtn=$('startCameraBtn'), stopBtn=$('stopCameraBtn'), switchBtn=$('switchCameraBtn');
  const fullscreenBtn=$('fullscreenBtn'), installBtn=$('installBtn'), detected=$('detectedSign');
  const confidenceValue=$('confidenceValue'), confidenceBar=$('confidenceBar'), translation=$('translationText');
  const typePill=$('typePill'), state=$('recognitionState'), toast=$('toast'), overlay=$('arOverlay');
  const arSign=$('arSign'), arMeaning=$('arMeaning'), rawOutput=$('rawOutput'), sentenceOutput=$('sentenceOutput');
  const speakBtn=$('speakBtn'), deleteBtn=$('deleteBtn'), clearBtn=$('clearSentenceBtn');
  const autoSpeak=$('autoSpeakToggle'), modelStatus=$('modelStatus');
  const navButtons=[...document.querySelectorAll('.nav-btn')], sections=[...document.querySelectorAll('.page-section')];

  const HOLISTIC_VERSION='0.5.1675471629';
  const HOLISTIC_BASE='https://cdn.jsdelivr.net/npm/@mediapipe/holistic@'+HOLISTIC_VERSION;
  const SEQUENCE_LENGTH=30, FEATURE_DIMENSION=456, SEQUENCE_SAMPLE_MS=1000/15, MIN_SEGMENT_FRAMES=8, QUIET_TO_END_MS=750, MAX_SEGMENT_FRAMES=180;
  const LABELS=['AKO','IKAW','SIYA','KAMI','OO','HINDI','KUMUSTA','SALAMAT','PAALAM','PAUMANHIN','PWEDE','GUSTO','AYAW','TULONG','MABUTI','MASAMA','ANO','SAAN','SINO','BAKIT'];
  const WORDS={AKO:'Ako',IKAW:'Ikaw',SIYA:'Siya',KAMI:'Kami',OO:'Oo',HINDI:'Hindi',KUMUSTA:'Kumusta',SALAMAT:'Salamat',PAALAM:'Paalam',PAUMANHIN:'Paumanhin',PWEDE:'Pwede',GUSTO:'Gusto',AYAW:'Ayaw',TULONG:'Tulong',MABUTI:'Mabuti',MASAMA:'Masama',ANO:'Ano',SAAN:'Saan',SINO:'Sino',BAKIT:'Bakit'};
  // Facial points emphasize mouth, brows, eyes and face outline for non-manual FSL features.
  const FACE_IDS=[10,152,234,454,1,4,33,133,159,145,362,263,386,374,61,291,13,14,70,300,105,334,78,308];
  const HAND_LINKS=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  const POSE_LINKS=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[29,31],[24,26],[26,28],[28,30],[30,32]];
  const FACE_LINES=[[33,160,158,133,153,144,33],[362,385,387,263,373,380,362],[61,146,91,181,84,17,314,405,321,375,291],[78,191,80,81,82,13,312,311,310,415,308],[70,63,105,66,107],[300,293,334,296,336]];

  let stream=null, holistic=null, facingMode='environment', rafId=null, processing=false, lastVideoTime=-1, lastFrameAt=0;
  let processEvery=1000/30, model=null, modelConfig=null, modelReady=false, modelLoadPromise=null;
  let segmentFrames=[], previousFeatures=null, signerVisible=false, inferenceBusy=false, lastInferenceAt=0;
  let averageInferenceMs=0, inferenceCount=0, inferenceEveryMs=100, predictionLabel='', predictionStreak=0, lastCommitAt=0;
  let segmentActive=false, segmentCommitted=false, lastMotionAt=0, speechTimer=null, sentence=[], lastSentenceLabel='';
  let installPrompt=null, voices=[];

  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const COMMIT_COOLDOWN_MS=700;
  const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
  const visible=p=>p&&Number.isFinite(p.x)&&Number.isFinite(p.y);
  function toastMessage(text){toast.textContent=text;toast.classList.add('show');clearTimeout(toastMessage.timer);toastMessage.timer=setTimeout(()=>toast.classList.remove('show'),2800)}
  function setState(text,cls=''){state.textContent=text;state.className=('recognition-state '+cls).trim()}
  function cameraStatus(text,online=false){status.classList.toggle('online',online);status.innerHTML='<span></span> '+text}
  function updateSentence(){rawOutput.textContent=sentence.length?sentence.join(' | '):'Wala pang na-decode na senyas.';sentenceOutput.textContent=sentence.length?sentence.map(w=>WORDS[w]||w).join(' '):'Ang na-decode na pangungusap ay lalabas dito.'}
  function stopSpeech(){if(speechTimer)clearTimeout(speechTimer);speechTimer=null;if('speechSynthesis'in window)window.speechSynthesis.cancel()}
  function chooseFilipinoVoice(){if(!('speechSynthesis'in window))return null;voices=window.speechSynthesis.getVoices()||[];return voices.find(v=>/^fil(-|$)/i.test(v.lang))||voices.find(v=>/^tl(-|$)/i.test(v.lang))||voices.find(v=>/^en-PH$/i.test(v.lang))||null}
  function speakSentence(){if(!autoSpeak?.checked||!sentence.length||!('speechSynthesis'in window))return;const utterance=new SpeechSynthesisUtterance(sentence.map(w=>WORDS[w]||w).join(' '));utterance.lang='fil-PH';utterance.rate=.92;const voice=chooseFilipinoVoice();if(voice)utterance.voice=voice;window.speechSynthesis.speak(utterance)}
  function scheduleSpeech(){if(speechTimer)clearTimeout(speechTimer);speechTimer=setTimeout(()=>{speechTimer=null;speakSentence()},1150)}
  function addWord(label,confidence){if(segmentCommitted)return;segmentCommitted=true;const now=performance.now();if(now-lastCommitAt<COMMIT_COOLDOWN_MS)return;sentence.push(label);lastSentenceLabel=label;lastCommitAt=now;updateSentence();toastMessage('Na-decode: '+label+' ('+Math.round(confidence)+'%)')}

  function showNoSign(message){detected.textContent='NO SIGN';confidenceValue.textContent='—';confidenceBar.style.width='0%';translation.textContent=modelReady?message:'Landmark tracking active; wala pang trained FSL model.';typePill.textContent=modelReady?'Waiting for sign':'Model not installed';setState(modelReady?'NO SIGN':'MODEL NOT READY');overlay.classList.remove('active')}
  function resizeCanvases(){const sourceW=video.videoWidth||video.clientWidth||1280,sourceH=video.videoHeight||video.clientHeight||720,s=Math.min(1,960/Math.max(sourceW,sourceH)),w=Math.max(1,Math.round(sourceW*s)),h=Math.max(1,Math.round(sourceH*s));for(const c of [handCanvas,poseCanvas])if(c&&(c.width!==w||c.height!==h)){c.width=w;c.height=h}}
  function mirrorPreview(){const mirror=facingMode==='user';video.classList.toggle('mirror',mirror);handCanvas.classList.toggle('mirror',mirror);poseCanvas?.classList.toggle('mirror',mirror)}
  function drawLinks(context,points,links,w,h,color,lineScale=2){if(!context||!points?.length)return;context.save();context.strokeStyle=color;context.lineWidth=Math.max(1.4,w/600*lineScale);context.lineCap='round';context.lineJoin='round';for(const [a,b] of links){if(!points[a]||!points[b])continue;context.beginPath();context.moveTo(points[a].x*w,points[a].y*h);context.lineTo(points[b].x*w,points[b].y*h);context.stroke()}context.restore()}
  function drawPoints(context,points,w,h,color,indices=null){if(!context||!points?.length)return;context.save();context.fillStyle=color;const list=indices||points.map((_,i)=>i);for(const i of list){const p=points[i];if(!p)continue;context.beginPath();context.arc(p.x*w,p.y*h,Math.max(1.4,w/850),0,Math.PI*2);context.fill()}context.restore()}
  function drawResults(r){resizeCanvases();const w=handCanvas.width,h=handCanvas.height;handCtx.clearRect(0,0,w,h);if(poseCtx)poseCtx.clearRect(0,0,w,h);const pose=r.poseLandmarks||[],left=r.leftHandLandmarks||[],right=r.rightHandLandmarks||[],face=r.faceLandmarks||[];
    drawLinks(handCtx,pose,POSE_LINKS,w,h,'rgba(56,189,248,.9)',2);drawPoints(handCtx,pose,w,h,'#bae6fd',[0,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]);
    drawLinks(handCtx,left,HAND_LINKS,w,h,'#a78bfa',2.7);drawPoints(handCtx,left,w,h,'#ddd6fe');
    drawLinks(handCtx,right,HAND_LINKS,w,h,'#34d399',2.7);drawPoints(handCtx,right,w,h,'#a7f3d0');
    const faceLinks=[];for(const line of FACE_LINES)for(let i=1;i<line.length;i++)faceLinks.push([line[i-1],line[i]]);
    drawLinks(poseCtx||handCtx,face,faceLinks,w,h,'rgba(251,191,36,.85)',1.2);drawPoints(poseCtx||handCtx,face,w,h,'#fde68a',FACE_IDS);
    const hands=Number(!!left.length)+Number(!!right.length);trackingBadge.textContent=pose.length?'BODY + FACE · '+hands+' HAND'+(hands===1?'':'S'):'TRACKING: SEARCHING';trackingBadge.classList.toggle('detected',!!pose.length);guide?.classList.toggle('is-hidden',!!pose.length);
  }

  function makeFeatures(r){return window.SignARFeatureExtractor?.extract(r,video.videoWidth||1280,video.videoHeight||720)||null}
  function measureMotion(v){if(!v||!previousFeatures)return 0;let sum=0;for(let i=0;i<384;i++)sum+=Math.abs(v[i]-previousFeatures[i]);return sum/384}

  async function loadModel(){if(modelLoadPromise)return modelLoadPromise;modelLoadPromise=(async()=>{
    try{const response=await fetch('./model/vocabulary.json',{cache:'no-store'});if(!response.ok)throw new Error('Vocabulary manifest is not present');modelConfig=await response.json();
      if(modelConfig.featureVersion!=='signar-fsl-456-v1')throw new Error('Feature schema mismatch; expected signar-fsl-456-v1');
      if(JSON.stringify(modelConfig.labels)!==JSON.stringify(LABELS))throw new Error('Vocabulary class order mismatch; expected the agreed 20-word order');
      if(modelConfig.sequenceLength!==SEQUENCE_LENGTH||modelConfig.featureDimension!==FEATURE_DIMENSION)throw new Error('Vocabulary input-shape contract mismatch');
      if(!modelConfig.trained||!modelConfig.modelPath){modelReady=false;modelStatus.textContent='Wala pang trained model. Landmark tracking works; sign predictions remain disabled until real FSL data is recorded and validated.';return}
      if(!window.tf)throw new Error('TensorFlow.js did not load');await window.tf.ready();model=await window.tf.loadLayersModel(modelConfig.modelPath);
      const input=model.inputs?.[0]?.shape||[],output=model.outputs?.[0]?.shape||[];
      if(input.length!==3||input[1]!==SEQUENCE_LENGTH||input[2]!==FEATURE_DIMENSION)throw new Error('Model input shape mismatch; expected [batch,30,456]');
      if(output.at(-1)!==LABELS.length)throw new Error('Model class count mismatch; expected 20 outputs');
      modelReady=true;modelStatus.textContent='Trained temporal model loaded. Prediction vocabulary contains 20 classes.';
    }catch(error){modelReady=false;console.info('FSL model unavailable:',error.message);modelStatus.textContent='Wala pang compatible na trained FSL model. Hindi gagawa ng pekeng prediction habang wala ito.'}
  })();return modelLoadPromise}

  async function infer(){if(!modelReady||!model||!segmentActive||inferenceBusy||segmentFrames.length<MIN_SEGMENT_FRAMES||performance.now()-lastInferenceAt<inferenceEveryMs)return;inferenceBusy=true;lastInferenceAt=performance.now();const started=performance.now();let input=null,output=null;
    try{const sampled=window.SignARFeatureExtractor.resampleSequence(segmentFrames,SEQUENCE_LENGTH);
      input=window.tf.tensor3d([sampled],[1,SEQUENCE_LENGTH,FEATURE_DIMENSION]);output=model.predict(input);const tensor=Array.isArray(output)?output[0]:output;const p=await tensor.data();let best=0;for(let i=1;i<p.length;i++)if(p[i]>p[best])best=i;
      const label=modelConfig.labels?.[best]||LABELS[best],confidence=clamp(p[best]*100,0,100);if(!label)return;
      if(predictionLabel===label)predictionStreak++;else{predictionLabel=label;predictionStreak=1}
      detected.textContent=label;confidenceValue.textContent=Math.round(confidence)+'%';confidenceBar.style.width=Math.round(confidence)+'%';translation.textContent='Prediction ng temporal model: '+label;typePill.textContent='TensorFlow.js sequence model';arSign.textContent=label;arMeaning.textContent=WORDS[label]||label;overlay.classList.add('active');setState('PREDICTING','verifying');
      // Never gate sentence insertion on confidence; segmenting and temporal stability prevent frame-by-frame duplicates.
      if(segmentActive&&!segmentCommitted&&predictionStreak>=2&&signerVisible)addWord(label,confidence);
    }catch(error){console.error('Inference error:',error);translation.textContent='Nagkaroon ng error sa model inference.';setState('MODEL ERROR')}
    finally{input?.dispose();if(output)(Array.isArray(output)?output:[output]).forEach(t=>t.dispose?.());const elapsed=performance.now()-started;inferenceCount++;averageInferenceMs+=(elapsed-averageInferenceMs)/inferenceCount;const inferenceDelay=averageInferenceMs>85?180:averageInferenceMs>50?130:100;inferenceEveryMs=inferenceDelay;inferenceBusy=false}
  }

  function onResults(r){drawResults(r);const vector=makeFeatures(r);signerVisible=!!vector;guide?.classList.toggle('is-hidden',signerVisible);
    if(!vector){segmentFrames=[];previousFeatures=null;predictionLabel='';predictionStreak=0;segmentActive=false;segmentCommitted=false;lastSentenceLabel='';showNoSign('Walang nakikitang katawan sa tracking area.');return}
    const now=performance.now(),energy=measureMotion(vector);
    if(energy>=.0055){lastMotionAt=now;if(!segmentActive){segmentActive=true;segmentCommitted=false;segmentFrames=[];predictionLabel='';predictionStreak=0;lastProcessedAtForSequence=0}stopSpeech()}
    previousFeatures=vector;
    if(segmentActive&&now-lastProcessedAtForSequence>=SEQUENCE_SAMPLE_MS){segmentFrames.push(vector);if(segmentFrames.length>MAX_SEGMENT_FRAMES)segmentFrames.shift();lastProcessedAtForSequence=now;void infer()}
    if(segmentActive&&segmentFrames.length>=MIN_SEGMENT_FRAMES&&now-lastMotionAt>QUIET_TO_END_MS&&(segmentCommitted||now-lastMotionAt>1400)&&!inferenceBusy){const hadPrediction=segmentCommitted;segmentActive=false;segmentCommitted=false;predictionLabel='';predictionStreak=0;segmentFrames=[];if(hadPrediction&&sentence.length)scheduleSpeech();showNoSign('Tapos na ang senyas.')}
    if(!modelReady)showNoSign('Landmark tracking active. Kailangan pa ang trained temporal model.');
  }
  let lastProcessedAtForSequence=0;

  async function initHolistic(){if(holistic)return;if(typeof window.Holistic!=='function')throw new Error('MediaPipe Holistic did not load');
    holistic=new window.Holistic({locateFile:file=>HOLISTIC_BASE+'/'+file});holistic.setOptions({modelComplexity:1,smoothLandmarks:true,refineFaceLandmarks:true,enableSegmentation:false,minDetectionConfidence:.5,minTrackingConfidence:.5});holistic.onResults(onResults);
    if(typeof holistic.initialize==='function')await holistic.initialize();await loadModel()
  }
  async function openCamera(){const ideal={width:{ideal:3840},height:{ideal:2160},frameRate:{ideal:30,max:60}};
    try{return await navigator.mediaDevices.getUserMedia({video:{...ideal,facingMode:{exact:facingMode}},audio:false})}
    catch(error){if(['OverconstrainedError','NotFoundError','TypeError'].includes(error?.name)){try{return await navigator.mediaDevices.getUserMedia({video:ideal,audio:false})}catch{return await navigator.mediaDevices.getUserMedia({video:true,audio:false})}}throw error}
  }
  function cameraError(e){if(!window.isSecureContext)return 'Camera access requires HTTPS or localhost.';if(e?.name==='NotAllowedError')return 'Pinayagan ang camera access? Payagan ang camera sa browser settings.';if(e?.name==='NotFoundError'||e?.name==='OverconstrainedError')return 'Walang compatible na camera sa device.';if(e?.name==='NotReadableError')return 'Maaaring ginagamit ng ibang app ang camera.';return 'Camera error: '+(e?.message||e?.name||'unknown')}
  async function startCamera(){try{if(!window.isSecureContext)throw new Error('HTTPS required');if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera API unavailable');await stopCamera(false);cameraStatus('Loading tracking models…');await initHolistic();stream=await openCamera();video.srcObject=stream;await video.play();
      const actual=stream.getVideoTracks()[0]?.getSettings?.().facingMode;if(actual)facingMode=actual;mirrorPreview();resizeCanvases();video.classList.add('active');placeholder.style.display='none';liveBadge.classList.add('show');trackingBadge.classList.add('show');startBtn.disabled=true;stopBtn.disabled=false;switchBtn.textContent=facingMode==='environment'?'Use front camera':'Use back camera';cameraStatus('Camera on',true);lastVideoTime=-1;lastFrameAt=0;rafId=requestAnimationFrame(frameLoop);
      toastMessage(modelReady?'Camera ready. Trained decoder loaded.':'Camera ready. Landmark tracking works; a trained model is still required.');
    }catch(e){console.error(e);await stopCamera(false);cameraStatus('Camera unavailable');translation.textContent=cameraError(e);toastMessage(cameraError(e))}}
  async function frameLoop(time){if(!stream)return;if(!processing&&video.readyState>=2&&video.currentTime!==lastVideoTime&&time-lastFrameAt>=processEvery){processing=true;lastFrameAt=time;lastVideoTime=video.currentTime;const started=performance.now();try{const sourceW=video.videoWidth||1280,sourceH=video.videoHeight||720,scale=Math.min(1,960/Math.max(sourceW,sourceH));processCanvas.width=Math.max(1,Math.round(sourceW*scale));processCanvas.height=Math.max(1,Math.round(sourceH*scale));processCtx.drawImage(video,0,0,processCanvas.width,processCanvas.height);await holistic.send({image:processCanvas})}catch(e){console.error(e);setState('TRACKING ERROR')}finally{const ms=performance.now()-started;processEvery=ms>75?1000/12:ms>48?1000/18:1000/30;processing=false}}rafId=requestAnimationFrame(frameLoop)}
  async function stopCamera(show=true){if(rafId)cancelAnimationFrame(rafId);rafId=null;processing=false;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;video.classList.remove('active');placeholder.style.display='flex';liveBadge.classList.remove('show');trackingBadge.classList.remove('show','detected');handCtx.clearRect(0,0,handCanvas.width,handCanvas.height);if(poseCtx)poseCtx.clearRect(0,0,poseCanvas.width,poseCanvas.height);startBtn.disabled=false;stopBtn.disabled=true;segmentFrames=[];previousFeatures=null;signerVisible=false;segmentActive=false;segmentCommitted=false;guide?.classList.remove('is-hidden');if(show){cameraStatus('Camera off');showNoSign('Walang kasalukuyang senyas.')}}
  async function switchCamera(){const desired=facingMode==='environment'?'user':'environment';facingMode=desired;if(stream){await stopCamera(false);await startCamera()}else{mirrorPreview();switchBtn.textContent=facingMode==='environment'?'Use front camera':'Use back camera'}}
  navButtons.forEach(btn=>btn.addEventListener('click',()=>{navButtons.forEach(b=>b.classList.toggle('active',b===btn));sections.forEach(s=>s.classList.toggle('active-section',s.id===btn.dataset.target))}));
  startBtn.addEventListener('click',startCamera);stopBtn.addEventListener('click',()=>stopCamera(true));switchBtn.addEventListener('click',switchCamera);
  fullscreenBtn.addEventListener('click',async()=>{try{if(!document.fullscreenElement)await cameraStage.requestFullscreen();else await document.exitFullscreen()}catch{toastMessage('Fullscreen is not available in this browser.')}});
  deleteBtn.addEventListener('click',()=>{sentence.pop();updateSentence();stopSpeech()});clearBtn.addEventListener('click',()=>{sentence=[];lastSentenceLabel='';updateSentence();stopSpeech()});speakBtn.addEventListener('click',()=>{stopSpeech();speakSentence()});
  window.addEventListener('resize',resizeCanvases);window.addEventListener('orientationchange',()=>setTimeout(resizeCanvases,250));window.addEventListener('beforeunload',()=>{void stopCamera(false);stopSpeech()});
  window.addEventListener('beforeinstallprompt',e=>{e.preventDefault();installPrompt=e;installBtn.hidden=false});installBtn.addEventListener('click',async()=>{if(!installPrompt)return;installPrompt.prompt();await installPrompt.userChoice;installPrompt=null;installBtn.hidden=true});
  if('speechSynthesis'in window)window.speechSynthesis.onvoiceschanged=chooseFilipinoVoice;
  if('serviceWorker'in navigator)window.addEventListener('load',()=>navigator.serviceWorker.register('./sw.js').catch(console.warn));
  switchBtn.textContent='Use front camera';mirrorPreview();updateSentence();showNoSign('Landmark tracking is ready. A trained FSL model is required for sign predictions.');
})();