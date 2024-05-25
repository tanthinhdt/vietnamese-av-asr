from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
import argparse

def args_parser():
    """
    Parse arguments.
    """

    parser = argparse.ArgumentParser(
        description="Download video from tiktok.")
    
    parser.add_argument('--channel_path',           type=str,
                        default="phdkhanh2507/testVLR/channel.txt",  help='Path list of channels (txt file) - 2 columns (channel_id, num_videos)')
    
    parser.add_argument('--save_path',           type=str,
                        default="phdkhanh2507/testVLR/video_url",  help='Path for saving channel')

    parser.add_argument('--overwrite',           action='store_true',
                        default=True,  help='Overwrite existing file')
    
    args = parser.parse_args()

    return args

def scroll_down_page(driver):
    """
    A method for scrolling the page.
    :param driver: selenium driver
    """

    scroll_pause_time = 2
    screen_height = driver.execute_script("return window.screen.height;")
    i = 0
    while True:
        driver.execute_script("window.scrollTo(0, {screen_height}*{i});".format(screen_height=screen_height, i=i))  
        i += 1
        time.sleep(scroll_pause_time)
        scroll_height = driver.execute_script("return document.body.scrollHeight;")  
        if (screen_height) * i > scroll_height:
            break 
        
def get_user_video(user_id, save_path, time_sleep=10):
    """
    Get video from user.
    :param user_id: user id(Ex: vietcetera)
    :param save_path: path for saving channel
    """

    options = Options()
    options.add_argument("start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver = webdriver.Chrome(options=options)

    # Change the tiktok link
    driver.get(f"https://www.tiktok.com/@{user_id}")

    # IF YOU GET A TIKTOK CAPTCHA, CHANGE THE TIMEOUT HERE
    # to 60 seconds, just enough time for you to complete the captcha yourself.
    time.sleep(time_sleep)

    # scroll down the page
    scroll_down_page(driver)


    # this class may change, so make sure to inspect the page and find the correct class
    className = " css-1as5cen-DivWrapper e1cg0wnj1"

    # Thay đổi script để chuyển NodeList thành một mảng trước khi sử dụng forEach
    script  = "let l = [];"
    script += "let elements = document.getElementsByClassName(\""
    script += className
    script += "\");"
    script += "for (let i = 0; i < elements.length; i++) {"
    script += "    let item = elements[i];"
    script += "    l.push(item.querySelector('a').href);"
    script += "}"
    script += "return l;"

    urlsToDownload = driver.execute_script(script)
    driver.close()
    # save the urls to a file
    if len(urlsToDownload) > 0:
        with open(os.path.join(save_path, f"{user_id}.txt"), "w") as f:
            for url in urlsToDownload:
                f.write(f"{url}\n")
    else:
        print(f"{user_id} has no videos")
    

if __name__ == "__main__":
    args = args_parser()

    # read channel list
    with open(args.channel_path, 'r') as f:
        lines = f.readlines()
    channels = [line.strip().split(',') for line in lines]        
    channels = [(channel[0], int(channel[1])) for channel in channels]
    
    for user_id, num_videos in channels:
        if (not os.path.exists(os.path.join(args.save_path, f"{user_id}.txt"))) or args.overwrite:
            try:
                get_user_video(user_id=user_id, save_path=args.save_path)
            except:
                print(f"Error in {user_id}")
                continue
        else:
            print(f"{user_id}.txt already exists")