# Vietnamese Image Captioning Service

----
## Docker
Build docker image:
```
docker build -t . <image_name>:<image_tag>
```
Run docker container:
```
docker run -d -p 5000:5000 --name <container_name> <image_name>:<image_tag>
```

----
## API Calling
* **URL**

    ```
    /api/image_captioning
    ```

* **Method:**

    `POST`   
*  **URL Params**

   **Required:**
 
   `image=[file]`

* **Success Response:**
  

  * **Code:** 200 <br />
    **Content:** 
    `{ 
        "message" :  "Successfully",
        "caption" : <Generated caption from the image>
        }`
 
* **Error Response:**

  * **Code:** 419 MISSING ARGUMENTS <br />
    **Content:** `{'message': 'No file selected'}`

  OR

  * **Code:** 420 INVALID ARGUMENTS <br />
    **Content:** `{'message': 'Not in allowed file'}`

* **Sample Call:**

    ```
    curl -F image=@<image_path> http://localhost:5000/api/image_captioning
    ``` 