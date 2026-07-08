from models import network


# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(GNet_opt):
    generator = getattr(network, GNet_opt.name)(GNet_opt.args)

    network.weights_init(generator, init_type = GNet_opt.init_type, init_gain = GNet_opt.init_gain)
    print('Generator is created!')
    if GNet_opt.finetune_path != "":
        generator.load_ckpt(GNet_opt.finetune_path, force_load = hasattr(GNet_opt, 'force_load') and GNet_opt.force_load)
        print('Generator is loaded!')
    return generator


def create_generator_val(GNet_opt, model_path = None, force_load = False):
    generator = getattr(network, GNet_opt.name)(GNet_opt.args)

    network.weights_init(generator, init_type = GNet_opt.init_type, init_gain = GNet_opt.init_gain)
    print('Generator is created!')

    if model_path is not None:
        generator.load_ckpt(model_path, force_load = force_load)
        print('Generator is loaded!')
    return generator

