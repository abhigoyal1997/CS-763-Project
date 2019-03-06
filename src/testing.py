from torch.utils.data import DataLoader


def test(model, dataset, model_path=None, predict_only=False):
    batch_size = 32
    num_workers = 4

    batches = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Test
    metrics = model.predict(batches)
    predictions = metrics.pop('predictions', None)  # TODO: save the predictions, maybe?
    if predict_only:
        return predictions

    print('Test: {}'.format(metrics))
