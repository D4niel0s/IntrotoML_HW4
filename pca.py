import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np

def plot_vector_as_image(image, h, w):
	"""
	utility function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimesnions of original pi
	"""
	plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
	plt.title('title', size=12)

def get_pictures_by_name(name='Ariel Sharon'):
	"""
	Given a name returns all the pictures of the person with this specific name.
	YOU CAN CHANGE THIS FUNCTION!
	THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
	"""
	lfw_people = load_data()
	selected_images = []
	n_samples, h, w = lfw_people.images.shape
	print("Got: ", list(lfw_people.target_names))
	target_label = list(lfw_people.target_names).index(name)
	for image, target in zip(lfw_people.images, lfw_people.target):
		if (target == target_label):
			image_vector = image.reshape((h*w, ))
			selected_images.append(image_vector)
	return selected_images, h, w

def load_data():
	# Don't change the resize factor!!!
	lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
	return lfw_people

######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""

def PCA(X, k):
	"""
	Compute PCA on the given matrix.

	Args:
		X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
		For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
		k - number of eigenvectors to return

	Returns:
	  U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
	  		of the covariance matrix.
	  S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
	"""
	n = X.shape[0]
	normX = X - np.mean(X, axis=0)
	
	U, S, Vh = np.linalg.svd(normX)
	
	S = S**2 #The eigenvalues of normX are the squares of the singular values

	indices = np.argpartition(S, -k)[-k:] #k largest eigenvalues / singular values
	
	U = Vh[indices]
	S = S[indices]

	return U,S
	

def main():
	images, h,w = get_pictures_by_name("George W Bush")
	X = np.array([images[i] for i in range(len(images))])

	partb(X, h,w)
	partc(X, h,w)

	plt.show()


	
def partb(X, h,w):
	U,S = PCA(X, 10)

	#Plot all Principle components
	fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
	fig.suptitle('Principal Components')
	for ax in axs:
		ax.remove()

	gridspec = axs[0].get_subplotspec().get_gridspec()
	subfigs = [fig.add_subfigure(gs) for gs in gridspec]

	for row, subfig in enumerate(subfigs):
		axs = subfig.subplots(nrows=1, ncols=5)
		for col, ax in enumerate(axs):
			ax.imshow(U[row*5 + col].reshape((h, w)), cmap=plt.cm.gray)
			ax.set_title(f'PC No.{row*5 + col + 1}')
			ax.axis("off")


def partc(X, h,w):
	indices = np.random.randint(0, X.shape[0], 5)
	ks = [1, 5, 10, 30, 50, 100]
	output = np.zeros((6,5,X.shape[1]))

	errs = np.zeros(6)

	i=0
	for k in ks:
		U,S = PCA(X, k)

		for img in X: #Sum up all errors
			errs[i] += np.linalg.norm(img - (U.T @ (U @ img)))

		for j in range(5): #Caculate the five random images
			output[i][j] = (U.T @ (U @ X[indices[j]]))

		i += 1
		

	#Plotting everything in a (kindof) nice format
	plt.figure(2)
	plt.title("PCA reconstruction error as a function of k")
	plt.plot(ks, errs, label="Reconstruction error", color="red")
	plt.xlabel("k")
	plt.legend()

	fig, axs = plt.subplots(nrows=7, ncols=1, constrained_layout=True)
	for ax in axs:
		ax.remove()

	gridspec = axs[0].get_subplotspec().get_gridspec()
	subfigs = [fig.add_subfigure(gs) for gs in gridspec]

	for row, subfig in enumerate(subfigs):
		if (row == 0):
			subfig.suptitle("Original images")
		else:
			subfig.suptitle(f"k={ks[row-1]}")
			
		axs = subfig.subplots(nrows=1, ncols=5)
		for col, ax in enumerate(axs):
			if(row == 0):
				ax.imshow(X[indices[col]].reshape((h, w)), cmap=plt.cm.gray)
				ax.axis("off")

			else:
				ax.imshow(output[row-1][col].reshape((h, w)), cmap=plt.cm.gray)
				ax.axis("off")

	


if __name__ == '__main__':
	main()