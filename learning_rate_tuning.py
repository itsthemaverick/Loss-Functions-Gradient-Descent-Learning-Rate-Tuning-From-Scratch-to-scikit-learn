def constant_lr(lr):
    return lr

def time_decay_lr(initial_lr,epoch,decay=0.01):
    return initial_lr/(1+decay*epoch)

def adaptive_lr(losses,lr):
    if len(losses) > 1 and losses[-1]>losses[-2]:
        return lr*0.5
    return lr
