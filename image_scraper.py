# This file is for the purpose of scraping images of Python snakes from google using Selenium

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import requests
import io
import os
from PIL import Image
import time

''' 
- In order to web scrape, we need to have a web driver so that 
there is a location (a browser) for the automation of operational processes
(- e.g. clicking and downloading images)
- without the web driver, our code will have no idea where to download the 
images from
'''
PATH = "/Users/rishvikkambhampati/Desktop/snake image classification/chromedriver"
service = Service(executable_path=PATH)
wd = webdriver.Chrome(service = service)

processed_urls = set()

def main():
    keywords = input("Enter the specific search terms: ") #kingsnakes, mexican black kingsnakes, california kingsnakes, scarlet kingsnakes, yellow-bellied kingsnakes
    data = keywords.split(", ")
    print(data)

    num = 0; 

    for keyword in data:
        image_nums = input("Enter the desired number of images: ")

        try:
            wd = webdriver.Chrome(service=service)
            url = "https://www.google.com/search?q=" + keyword + "&sca_esv=e23fa3d6ecc2111f&udm=2&biw=1200&bih=673&sxsrf=ADLYWIKLoOxqvgt3WyaeQ408oFviWQEJ6w%3A1732575467632&ei=6wBFZ9WcJuDakPIP996K2Qc&ved=0ahUKEwiVvq3KyviJAxVgLUQIHXevInsQ4dUDCBA&uact=5&oq=kingsnake&gs_lp=EgNpbWciCWtpbmdzbmFrZTIEECMYJzIKEAAYgAQYQxiKBTIFEAAYgAQyBRAAGIAEMgUQABiABDIKEAAYgAQYQxiKBTIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgARIvAtQjwRYjwRwAXgAkAEAmAEsoAEsqgEBMbgBA8gBAPgBAZgCAqACNMICBhAAGAcYHpgDAIgGAZIHATKgB_4F&sclient=img"
            img_urls = download_google_images(wd, 3, int(image_nums), url)
            for i, img_url in enumerate(img_urls):
                download_image("/Users/rishvikkambhampati/Desktop/snake image classification/images/", img_url, str(num) + ".jpg")
                num += 1

            wd.quit()

        except Exception as e: 
            print(f"Error processing keyword {keyword}: {e}")




'''
- a method to automate the process of downloading a bunch of images by clicking on the thumbnail, 
grabbing the image source, and then downloading the image from its img source
'''
def download_google_images(webdriver, delay, max_images, url):
    def scroll_down(webdriver):
        last_height = webdriver.execute_script("return document.body.scrollHeight")
        while True:
            webdriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")    # The driver scrolls down the webpage here.
            time.sleep(8)
            new_height = webdriver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:   # Breaks here if the driver reached the bottom of the webpage (when it cannot scroll down anymore).
                break
            last_height = new_height        # If the driver failed to scroll to the bottom of the webpage, the current scroll height is recorded. Following, it is compared to the scroll height in the next iteration of the loop to decide if the driver reached the bottom of the webpage or not.
        webdriver.execute_script("window.scrollTo(0, 0);")

    webdriver.get(url)

    img_urls = set() #making it a set so that there's no repeats of images
    processed_indices = set()
    

    while len(img_urls) < max_images:

        scroll_down(webdriver)
        thumbnails = webdriver.find_elements(By.CSS_SELECTOR, ".YQ4gaf:not([class*=' '])")
        
        #for tn in thumbnails[len(img_urls): max_images]:
        for idx, tn in enumerate(thumbnails):
            print("Loop " + str(len(img_urls)))
            '''if there's an error with clicking the images, we don't want to interrupt 
                the script, so we just continue to clicking the next image we find'''
            if len(img_urls) >= max_images:
                break

            if idx in processed_indices:
                continue

            processed_indices.add(idx)

            try: 
                WebDriverWait(webdriver, 10).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "YQ4gaf"))
                )
                tn.click()
                time.sleep(delay)
            except Exception as e:
                print(f"Skipping problematic thumbnail due to error: {e}")
                continue

            try: 
                images = WebDriverWait(webdriver, 10).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "FyHeAf"))
                ) #iPVvYb
            except Exception as e:
                print(f"Skipping problematic image due to error: {e}")
                continue

            #images = webdriver.find_elements(By.CLASS_NAME, "iPVvYb")
            for img in images:

                if img.get_attribute('src') in img_urls:
                    continue
                
                if img.get_attribute('src') and 'http' in img.get_attribute('src') and not img.get_attribute('src').startswith("https://encrypted-tbn"):
                    if img.get_attribute('src') not in processed_urls:
                        img_urls.add(img.get_attribute('src'))
                        processed_urls.add(img.get_attribute('src'))
                        print(f"Image located {len(img_urls)}")

    return img_urls
            



# A method to indirectly download an image from a browser
def download_image(download_path, url, file_name):
    '''
    - for the image content, we are making an HTTP request to the given url
    and extracting the content of the url, which in this case is the image
    - then, we convert the image into a BytesIO file type to make the process of 
    performing operations on the image more efficient; this BytesIO file type is directly 
    converted into an actual image that can be opened through the Pillar library
    - lastly, we download the image using the with open() function
    '''
    try:
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content) # BytesIO is a virtual file that simply exists in the computer's memory
        image = Image.open(image_file)
        file_path = os.path.join(download_path, file_name)

        with open(file_path, "wb") as f:
            image.save(f, "JPEG")

        print("Success")
    except Exception as e:
        print('failed', e)


main()