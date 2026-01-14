from .resnet import *
import logging
logger = logging.getLogger('base')

def create_CD_model(opt):
    cd_model = None
    model_name = opt['model']['name']

    # Lazy import: only import the model that's actually requested
    if model_name == 'cdmamba':
        from models.CDMamba import CDMamba as cdmamba
        cd_model = cdmamba(spatial_dims=opt['model']['spatial_dims'], in_channels=opt['model']['in_channels'], init_filters=opt['model']['init_filters'], num_classes=opt['model']['n_classes'],
                              mode=opt['model']['mode'], conv_mode=opt['model']['conv_mode'], up_mode=opt['model']['up_mode'], up_conv_mode=opt['model']['up_conv_mode'], norm=opt['model']['norm'],
                              blocks_down=opt['model']['blocks_down'], blocks_up=opt['model']['blocks_up'], resdiual=opt['model']['resdiual'], diff_abs=opt['model']['diff_abs'], stage=opt['model']['stage'],
                              mamba_act=opt['model']['mamba_act'], local_query_model=opt['model']['local_query_model'])
    elif model_name == 'cdmamba_modified':
        from models.CDMamba_modified import CDMamba as cdmamba_modified
        cd_model = cdmamba_modified(spatial_dims=opt['model']['spatial_dims'], in_channels=opt['model']['in_channels'], init_filters=opt['model']['init_filters'], num_classes=opt['model']['n_classes'],
                              mode=opt['model']['mode'], conv_mode=opt['model']['conv_mode'], up_mode=opt['model']['up_mode'], up_conv_mode=opt['model']['up_conv_mode'], norm=opt['model']['norm'],
                              blocks_down=opt['model']['blocks_down'], blocks_up=opt['model']['blocks_up'], resdiual=opt['model']['resdiual'], diff_abs=opt['model']['diff_abs'], stage=opt['model']['stage'],
                              mamba_act=opt['model']['mamba_act'], local_query_model=opt['model']['local_query_model'])
    elif model_name in ['cdmamba_seg', 'CDMamba_seg']:
        from models.CDMamba_Segmentation import CDMamba_seg as cdmamba_seg
        cd_model = cdmamba_seg(
            spatial_dims=opt['model']['spatial_dims'],
            in_channels=opt['model']['in_channels'],
            init_filters=opt['model']['init_filters'],
            num_classes=opt['model']['n_classes'],
            conv_mode=opt['model']['conv_mode'],
            norm=opt['model']['norm'],
            blocks_down=opt['model']['blocks_down'],
            blocks_up=opt['model']['blocks_up'],
            up_conv_mode=opt['model']['up_conv_mode'],
        )
    elif model_name in ['cdmamba_seg_cd', 'CDMamba_seg_cd']:
        from models.CDMamba_Seg_change import CDMamba_seg_cd as cdmamba_seg_cd
        cd_model = cdmamba_seg_cd(
            spatial_dims=opt['model']['spatial_dims'],
            in_channels=opt['model']['in_channels'],
            init_filters=opt['model']['init_filters'],
            num_classes=opt['model']['n_classes'],
            use_change_head=opt['model'].get('use_change_head', True),  # Default to True if not specified
            conv_mode=opt['model']['conv_mode'],
            norm=opt['model']['norm'],
            blocks_down=opt['model']['blocks_down'],
            blocks_up=opt['model']['blocks_up'],
            up_conv_mode=opt['model']['up_conv_mode'],
        )
    elif model_name == 'bifa':
        from models.bifa import BiFA as bifa
        cd_model = bifa(backbone="mit_b0")
    elif model_name == 'video_bcd':
        from models.swin3d import Video_Bcd as video_bcd
        cd_model = video_bcd(video_len=opt['model']['video_len'], num_cls=2, mode=opt['model']['mode'])
    elif model_name == 'changemamba':
        from models.mamba_cd import STMambaBCD as changemamba
        cd_model = changemamba(pretrained="", patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], dims=96,
                          ssm_d_state=16, ssm_ratio=2.0, ssm_rank_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
                          ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0", forward_type="v2",
                          mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, drop_path_rate=0.1, patch_norm=True,
                          norm_layer='ln', downsample_version="v2", patchembed_version="v2", gmlp=False,
                          use_checkpoint=False, device=opt['model']['device'])
    elif model_name == 'rs_cdmamba':
        from models.rs_mamba import RSM_CD as rs_cdmamba
        cd_model = rs_cdmamba(drop_path_rate=0.2, dims=96, depths=[ 2, 2, 9, 2 ], ssm_d_state=16, ssm_dt_rank="auto",
                      ssm_ratio=2.0, mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2",
                      image_size=256, downsample_raito=1)

    elif model_name == 'bit':
        from models.bit import BASE_Transformer as bit
        cd_model = bit(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                     with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
        print("bit")
    elif model_name == 'mscanet':
        from models.mscanet import MSCACDNet as mscanet
        cd_model = mscanet()
        print("mscanet")
    elif model_name == 'changeformer':
        from models.changeformer import ChangeFormerPre as changeformer
        cd_model = changeformer()
        print("changeformer")
    elif model_name == 'paformer':
        from models.paformer import Paformer as paformer
        cd_model = paformer()
        print("paformer")
    elif model_name == 'darnet':
        from models.darnet import DARNet as darnet
        cd_model = darnet()
        print("darnet")
    elif model_name == 'snunet':
        from models.snunet import SiamUnet_diff as snunet
        cd_model = snunet(input_nbr=3, label_nbr=2)
        print("snunet")
    elif model_name == 'ifnet':
        from models.ifnet import DSIFN as ifnet
        cd_model = ifnet()
        print("ifnet")
    elif model_name == 'dminet':
        from models.dminet import DMINet as dminet
        cd_model = dminet()
        print("dminet")
    elif model_name == 'fc_ef':
        from models.fc_ef import UNet as fc_ef
        cd_model = fc_ef(in_ch=6, out_ch=2)
        print("fc_ef")
    elif model_name == 'fc_siam_conc':
        from models.fc_siam_conc import SiamUNet_conc as fc_siam_conc
        cd_model = fc_siam_conc(in_ch=3, out_ch=2)
        print("fc_siam_conc")
    elif model_name == 'fc_siam_diff':
        from models.fc_sima_diff import SiamUNet_diff as fc_siam_diff
        cd_model = fc_siam_diff(in_ch=3, out_ch=2)
        print("fc_siam_diff")
    elif model_name == 'acabfnet':
        from models.acabfnet import CrossNet as acabfnet
        cd_model = acabfnet(nclass=2, head=[4,8,16,32])
        print("acabfnet")
    else:
        # Unknown model name
        print("No model")
    
    if cd_model is None:
        raise ValueError(f"Unknown model name: {opt['model']['name']}")
    
    logger.info('CD Model [{:s}] is created.'.format(opt['model']['name']))
    return cd_model