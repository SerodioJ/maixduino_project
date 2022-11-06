def update_sample(request):
    sample = request.app.state.sample + 1
    request.app.state.sample = sample
    return sample
