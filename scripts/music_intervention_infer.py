"""Step C: released inference with fixed codes and explicit music intervention."""
from pathlib import Path
import sys, json, argparse, subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import smoke_infer as smoke


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case', choices=['A-repeat','B-swap','A-shift'], required=True)
    p.add_argument('--output', required=True)
    a=p.parse_args()
    import numpy as np
    cfg=json.loads(Path('/root/project/configs/smoke_batch01.json').read_text())
    cfg['policy']['path']='/vol/assets/policy/epoch_400.pt'
    cfg['position_decoder']['path']='/vol/assets/decoder/epoch_500.pt'
    cfg['motion_record']='/vol/assets/motion/gHO_sBM_cAll_d20_mHO5_ch01.json'
    cfg['external_features']='/vol/assets/features/mHO5.json'
    out=Path(a.output); out.mkdir(parents=True, exist_ok=True)
    song='mJB5' if a.case=='B-swap' else 'mHO5'
    shift=8 if a.case=='A-shift' else 0
    source=Path('/vol/assets/features')/(song+'.json')
    values=np.asarray(json.loads(source.read_text())['music_array'])
    original_stage=smoke.stage_example
    def stage(config, root, work):
        result=original_stage(config,root,work)
        # Keep exactly the released smoke padding; shift valid source values only.
        target=result['music'].copy()
        selected=values[shift:shift+1200]
        assert selected.shape==(1200,438)
        target[40:1240]=selected
        target[1240]=selected[-1]
        assert np.array_equal(target[40:1240],values[np.arange(1200)+shift])
        assert not target[:40].any() and not target[1241:].any()
        result['music']=target
        return result
    smoke.stage_example=stage
    original_encode=smoke.encode_first
    def encode(decoder, pose):
        initial,full=original_encode(decoder,pose)
        initial[0].fill_(436); initial[1].fill_(289)
        return initial,full
    smoke.encode_first=encode
    cfg['audio']='/vol/assets/audio/'+song+'.wav'
    if shift:
        shifted=out/'audio_shifted.wav'
        subprocess.run(['ffmpeg','-v','error','-y','-i',cfg['audio'],'-af','atrim=start_sample=6400,asetpts=PTS-STARTPTS','-c:a','pcm_s24le',str(shifted)],check=True)
        cfg['audio']=str(shifted)
    config_path=out/'resolved_config.json'; config_path.write_text(json.dumps(cfg,indent=2))
    intervention={'case':a.case,'conditioning_song':song,'feature_advance_frames':shift,'audio_advance_samples_48000hz':shift*800,'fixed_initial_codes':[436,289],'source_feature_sha256':smoke.sha256_file(source),'timestamp_check':'conditioning frame t uses source t+advance; no wrap','boundary_policy':'identical 40-frame zero padding to A-original; score only [5,15) s'}
    (out/'intervention.json').write_text(json.dumps(intervention,indent=2))
    smoke.run(argparse.Namespace(config=str(config_path),project='/root/project',output=str(out),device='cuda',load_only=False,runtime_label='legacy'))

if __name__=='__main__': main()
