
import cv2
import random, string
import numpy as np



viusalizeInterpolation(X_test[randint(0,X_test.shape[0])], X_test[randint(0,X_test.shape[0])], generator, discriminator)

# Shows linear inteprolation in image space vs latent space
def visualizeInterpolation(start, end, generator, discriminator, nbSteps=10):
    print("Generating interpolations...")
    generator.eval()
    # Create micro batch
    X = Variable(torch.randn((inputs.size(0), 100)).view(-1, 100, 1, 1))

    # Compute latent space projection
    latentX = encoder.predict(X)
    latentStart, latentEnd = latentX

    # Get original image for comparison
    startImage, endImage = X

    vectors = []
    normalImages = []
    # Linear interpolation
    alphaValues = np.linspace(0, 1, nbSteps)
    for alpha in alphaValues:
        # Latent space interpolation
        vector = latentStart * (1 - alpha) + latentEnd * alpha
        vectors.append(vector)
        # Image space interpolation
        blendImage = cv2.addWeighted(startImage, 1 - alpha, endImage, alpha, 0)
        normalImages.append(blendImage)

    # Decode latent space vectors
    vectors = np.array(vectors)
    reconstructions = generator(vectors.view(-1, 100, 1, 1))

    # Put final image together
    resultLatent = None
    resultImage = None

    hashName = ''.join(random.choice(string.lowercase) for i in range(3))

    for i in range(len(reconstructions)):
        interpolatedImage = normalImages[i] * 255
        interpolatedImage = cv2.resize(interpolatedImage, (50, 50))
        interpolatedImage = interpolatedImage.astype(np.uint8)
        resultImage = interpolatedImage if resultImage is None else np.hstack([resultImage, interpolatedImage])

        reconstructedImage = reconstructions[i] * 255.
        reconstructedImage = reconstructedImage.reshape([28, 28])
        reconstructedImage = cv2.resize(reconstructedImage, (50, 50))
        reconstructedImage = reconstructedImage.astype(np.uint8)
        resultLatent = reconstructedImage if resultLatent is None else np.hstack([resultLatent, reconstructedImage])

        cv2.imwrite('logs' + "{}_{}.png".format(hashName, i),
                    np.hstack([interpolatedImage, reconstructedImage]))
