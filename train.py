import torch
import math
import time


def collate(data):
    images = torch.stack([item[0] for item in data], dim=0)
    labels = torch.tensor([[item[1]] for item in data])
    return images, labels


def train(model, optimizer, lossFunction, dataset, device, epochs=50, batchSize=8):

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batchSize,
        shuffle=True,
        collate_fn=collate
    )

    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
        print()

        lastSeenProgressProcent = -1
        correctAnswers = 0
        runningLoss = 0
        numberOfPredictions = 0
        since = time.time()

        for index, samples in enumerate(dataloader):
            images, labels = samples

            images = images.to(device)
            labels = labels.to(device)

            numberOfPredictions += len(images)

            optimizer.zero_grad()

            prediction = model(images)

            loss = lossFunction(prediction, labels)
            loss.backward()

            prediction = prediction >= 0.5
            prediction = prediction.float()

            correctAnswers += torch.sum(prediction == labels).item()
            runningLoss += loss.item()

            optimizer.step()

            lastSeenProgressProcent = printProgress(
                index, batchSize, len(dataset), lastSeenProgressProcent)

        print()
        print()

        stats = calculateEpochStats(
            runningLoss, correctAnswers, numberOfPredictions, since)
        printStats(stats)

        print()

    return model


def printProgress(index, batchSize, datasetLength, lastSeenProgressProcent):
    progress = ((index+1)*batchSize)/(datasetLength)
    progressProcent = math.floor(progress * 100)

    if progressProcent >= lastSeenProgressProcent + 1:
        print("\rProgress: {}%".format(progressProcent), end="")
        lastSeenProgressProcent = progressProcent
    return lastSeenProgressProcent


def calculateEpochStats(loss, correctAnswers, numberOfPredictions, since):
    epochTime = time.time() - since
    averageLoss = loss / numberOfPredictions
    accuracy = correctAnswers / numberOfPredictions

    return {
        "duration": epochTime,
        "loss": averageLoss,
        "accuracy": accuracy
    }


def printStats(stats):
    print('Loss: {:.4f} Accuracy: {:.4f} Duration: {:.0f}m {:.0f}s'.format(
        stats["loss"], stats["accuracy"], stats["duration"] // 60, stats["duration"] % 60))
