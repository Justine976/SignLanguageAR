/* Shared SignAR FSL feature extraction. Keep this schema identical for collection, training, and inference. */
(() => {
  'use strict';
  const DIMENSION=456;
  const FACE_IDS=[10,152,234,454,1,4,33,133,159,145,362,263,386,374,61,291,13,14,70,300,105,334,78,308];
  const visible=p=>p&&Number.isFinite(p.x)&&Number.isFinite(p.y);
  const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  function extract(results,width=1280,height=720){
    const pose=results.poseLandmarks||[],face=results.faceLandmarks||[],left=results.leftHandLandmarks||[],right=results.rightHandLandmarks||[];
    if(pose.length<33||![11,12,23,24].every(i=>visible(pose[i])&&(pose[i].visibility??1)>=.25))return null;
    const shoulder={x:(pose[11].x+pose[12].x)/2,y:(pose[11].y+pose[12].y)/2};
    const hip={x:(pose[23].x+pose[24].x)/2,y:(pose[23].y+pose[24].y)/2};
    const anchorX=(shoulder.x+hip.x)*.5*width,anchorY=(shoulder.y+hip.y)*.5*height;
    const scale=Math.max(1,Math.hypot((hip.x-shoulder.x)*width,(hip.y-shoulder.y)*height));
    const norm=p=>visible(p)?[(p.x*width-anchorX)/scale,(p.y*height-anchorY)/scale,((p.z||0)*width)/scale]:[0,0,0];
    const vector=[];
    for(let i=0;i<33;i++){const p=pose[i],n=norm(p);vector.push(...n,clamp(p?.visibility??0,0,1))}
    for(const hand of [left,right]){
      const wrist=hand?.[0];
      const palm=hand?.length>17?Math.max(1,Math.hypot((hand[5].x-hand[17].x)*width,(hand[5].y-hand[17].y)*height)):1;
      for(let i=0;i<21;i++){const p=hand?.[i],g=norm(p),local=p&&wrist?[(p.x-wrist.x)*width/palm,(p.y-wrist.y)*height/palm,((p.z||0)-(wrist.z||0))*width/palm]:[0,0,0];vector.push(...g,...local)}
    }
    for(const i of FACE_IDS)vector.push(...norm(face[i]));
    while(vector.length<DIMENSION)vector.push(0);
    return vector.slice(0,DIMENSION);
  }
  function resampleSequence(frames,target=30){
    if(!Array.isArray(frames)||frames.length<2||!Number.isInteger(target)||target<2)throw new Error('A sequence needs at least two frames and a target length >= 2');
    if(frames.some(frame=>!Array.isArray(frame)||frame.length!==DIMENSION||frame.some(value=>!Number.isFinite(value))))throw new Error('Invalid landmark feature frame');
    const last=frames.length-1,out=[];
    for(let i=0;i<target;i++){
      const position=i*last/(target-1),lo=Math.floor(position),hi=Math.min(lo+1,last),weight=position-lo;
      const frame=new Array(DIMENSION);
      for(let j=0;j<DIMENSION;j++)frame[j]=frames[lo][j]*(1-weight)+frames[hi][j]*weight;
      out.push(frame);
    }
    return out;
  }
  window.SignARFeatureExtractor=Object.freeze({version:'signar-fsl-456-v1',dimension:DIMENSION,samplingFps:15,sequenceLength:30,faceIds:FACE_IDS,extract,resampleSequence});
})();
