
import cv2
import numpy as np
 
MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.08
 
def alignImages(im1, im2):
 

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  im1_display = cv2.drawKeypoints(im1, keypoints1, outImage=np.array([]), color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  im2_display = cv2.drawKeypoints(im2, keypoints2, outImage=np.array([]), color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_L1)
  matches = matcher.match(descriptors1, descriptors2, None)
  print("originalni mecevi/n")
  no_of_matches = sorted(matches,key=lambda x:x.distance)
  print("POKLAPAJUCI mecevi/n")

  numGoodMatches = int(len(no_of_matches) * GOOD_MATCH_PERCENT)
  no_of_matches = no_of_matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, no_of_matches, None)
  cv2.imwrite("gmatch.jpg", imMatches)

  # Extract location of good matches
  points1 = np.zeros((len(no_of_matches), 2), dtype=np.float32)
  points2 = np.zeros((len(no_of_matches), 2), dtype=np.float32)
 
  for i, match in enumerate(no_of_matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
 
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
 
  return im1Reg, h

def uskladi(img1):
       down_width = 600
       down_height = 900
       down_points = (down_width, down_height)
       
       resized_down = cv2.resize(img1, down_points, interpolation= cv2.INTER_LINEAR)
       return resized_down
       

if __name__ == '__main__':
 
  # Read reference image
  refFilename = "im1.jpg"
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
 
  # Read image to be aligned
  imFilename = "im2.jpg"
  print("Reading image to align : ", imFilename);
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

  #s1=uskladi(im)
  #s_ref=uskladi(imReference)
 
  print("Aligning images ...")
 
  
  imReg, h = alignImages(im, imReference)
  # Write aligned image to disk.
  outFilename = "gligned.jpg"
  print("Saving aligned image : ", outFilename);
  cv2.imwrite(outFilename, imReg)
  cv2.imshow("rezultat",imReg )
  cv2.waitKey(0)
  print("Estimated homography : \n",  h)