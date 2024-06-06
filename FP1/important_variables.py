import base64


img_width, img_height = 512, 512

input_shape = (img_width, img_height)


theme_image_name = 'theme_image.png'
logo_image = 'theme_image.png'

"""### gif from local file"""
file_ = open(theme_image_name, "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()


css_file_path = 'style/style.css'


