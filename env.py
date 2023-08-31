container_name = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "DenseNet121", "DenseNet121",    "ResNet18", "ResNet34", "ResNet50", "ResNet101", "DenseNet121", "DenseNet121",    "ResNet18", "ResNet34", "ResNet50", "ResNet101", "DenseNet121", "DenseNet121"]
input_size = [224*224*3, 224*224*3, 112*112*3, 112*112*3, 112*112*3, 224*224*3,    224*224*3, 224*224*3, 112*112*3, 112*112*3, 112*112*3, 224*224*3,     224*224*3, 224*224*3, 112*112*3, 112*112*3, 112*112*3, 224*224*3]
# workload 运行所需要的G flops
workload = [0.969, 4.134, 2.022, 2.957, 0.718, 2.897,    0.969, 4.134, 2.022, 2.957, 0.718, 2.897,    0.969, 4.134, 2.022, 2.957, 0.718, 2.897]
accuracy = [8.58, 7.13, 6.44+2, 5.94+2, 7.83+2, 7.83,    8.58, 7.13, 6.44+2, 5.94+2, 7.83+2, 7.83,    8.58, 7.13, 6.44+2, 5.94+2, 7.83+2, 7.83]
# inference_time_edge 模型在edge（xavier AGX）上的运行时间,单位ms
inference_time_edge = [
    [20.17, 19.25, 20.94, 24.3, 25.84, 29.78, 28.21, 30.54, 31.61, 41.18, 44.74, 44.65, 40.9, 43.63, 40.09, 41.38, 54.23, 55.68, 57.91, 59.39, 61.56, 63.41, 64.26, 64.98, 66.2, 67.55, 69.28, 70.22, 71.77, 72.32, 74.11, 76.19, 97.25, 98.16, 98.99, 100.12, 101.05, 102.37, 102.91, 105.1, 106.31, 108.19, 108.78, 109.81, 118.74, 119.16, 120.84, 121.86, 130.21, 148.7, 130.03, 131.06, 132.96, 133.46, 135.61, 136.3, 137.62, 138.95, 139.26, 140.56, 142.34, 143.27, 144.83, 147.36, 167.26, 169.17, 169.57, 170.83, 172.04, 172.97, 174.4, 175.42, 176.16, 178.04, 179.11, 180.43, 181.97, 182.97, 182.8, 184.56, 198.77, 199.79, 199.83, 202.78, 204.25, 206.27, 208.55, 210.42, 212.73, 215.12, 216.57, 218.35, 219.79, 221.6, 223.94, 225.92, 242.88, 244.53, 246.3, 248.48],
    [34.88, 32.84, 33.67, 48.66, 51.85, 58.59, 52.21, 55.81, 57.01, 73.83, 78.67, 78.02, 69.23, 73.01, 68.44, 70.5, 99.5, 101.57, 104.2, 106.28, 109.11, 109.96, 111.42, 112.89, 114.43, 115.64, 118.12, 119.65, 120.99, 122.49, 123.72, 127.04, 168.82, 169.96, 171.4, 172.79, 174.48, 175.52, 176.92, 179.64, 180.52, 182.77, 184.57, 185.68, 196.61, 197.7, 199.2, 200.69, 222.88, 252.99, 225.09, 226.48, 228.33, 230.05, 232.07, 233.07, 234.72, 235.94, 237.52, 238.59, 241.09, 242.39, 244.51, 247.02, 287.99, 289.57, 290.75, 292.02, 293.31, 294.68, 296.96, 298.82, 299.84, 301.11, 302.88, 305.15, 306.7, 308.28, 310.12, 311.42, 341.73, 343.56, 343.68, 347.57, 351.24, 353.79, 357.22, 359.22, 362.76, 365.21, 367.23, 370.21, 372.27, 375.96, 378.39, 381.01, 415.89, 418.29, 420.54, 423.38],
    [36.55, 36.16, 36.49, 33.96, 31.85, 31.23, 32.19, 33.74, 37.23, 39.15, 41.05, 41.63, 38.78, 41.06, 46.93, 48.23, 54.83, 55.28, 57.43, 58.32, 61.24, 60.76, 61.66, 63.74, 66.52, 67.3, 72.08, 73.08, 75.15, 78.63, 81.06, 84.62, 93.33, 91.37, 93.11, 93.86, 95.91, 96.93, 98.06, 100.95, 103.09, 106.6, 107.76, 229.74, 232.48, 230.74, 231.71, 236.44, 238.82, 123.06, 120.95, 124.06, 125.81, 126.96, 133.24, 134.55, 136.86, 138.78, 141.26, 142.56, 144.92, 145.87, 149.86, 152.33, 159.39, 161.23, 158.38, 160.49, 161.49, 163.17, 165.04, 166.24, 168.9, 171.72, 173.05, 180.43, 181.66, 182.97, 178.85, 180.25, 185.93, 187.7, 203.35, 206.17, 208.11, 209.42, 210.9, 211.83, 215.34, 217.24, 215.82, 216.73, 218.71, 220.16, 222.86, 223.41, 238.65, 238.99, 241.29, 244.38],
    [67.63, 67.54, 70.04, 64.34, 64.57, 65.22, 59.98, 61.63, 66.32, 67.76, 74.41, 72.58, 72.78, 73.79, 76.49, 83.31, 91.27, 92.76, 96.22, 97.63, 102.38, 102.26, 103.11, 107.08, 109.5, 111.22, 118.8, 120.04, 123.41, 132.76, 136.28, 142.98, 161.88, 155.02, 156.24, 157.86, 162.38, 163.8, 165.31, 168.6, 172.59, 178.81, 180.82, 302.42, 305.4, 305.17, 306.3, 312.3, 315.39, 199.85, 197.84, 200.79, 204.38, 205.98, 213.13, 214.47, 218.11, 221.42, 223.41, 225.11, 228.7, 230.06, 238.82, 240.89, 257.69, 260.91, 277.29, 281.28, 264.47, 265.96, 270.36, 271.83, 274.46, 279.96, 281.46, 291.13, 292.73, 294.49, 290.11, 292.37, 300.36, 302.5, 318.97, 325.04, 326.39, 328.63, 331.85, 333.33, 339.25, 340.56, 342.45, 344.82, 346.76, 348.7, 355.16, 356.76, 381.08, 381.23, 384.29, 389.57],
    [55.82, 54.23, 55.02, 56.16, 57.02, 56.49, 51.39, 51.43, 51.06, 54.5, 55.4, 53.97, 55.43, 54.73, 56.55, 54.02, 53.45, 54.06, 54.57, 55.61, 57.92, 58.71, 60.53, 62.62, 65.3, 66.63, 69.93, 71.29, 73.05, 74.78, 76.41, 79.88, 84.0, 83.74, 84.38, 86.08, 87.48, 89.6, 90.8, 92.97, 94.49, 99.09, 100.66, 102.12, 103.61, 105.72, 106.82, 109.12, 113.35, 114.81, 116.35, 118.56, 120.55, 121.14, 126.58, 127.9, 130.23, 132.1, 133.5, 135.61, 139.33, 141.0, 144.5, 145.71, 149.05, 150.08, 152.56, 155.02, 156.86, 158.13, 159.85, 161.89, 163.9, 167.46, 169.06, 174.33, 175.88, 178.01, 174.42, 175.25, 184.4, 186.45, 202.77, 206.15, 208.28, 209.57, 211.1, 211.94, 216.02, 217.16, 228.95, 230.17, 232.88, 234.83, 238.23, 239.61, 248.33, 249.9, 250.59, 254.5],
    [93.04, 84.16, 80.98, 84.27, 83.05, 85.69, 83.52, 88.34, 92.28, 96.65, 104.7, 111.97, 110.47, 117.12, 123.7, 133.1, 146.05, 152.31, 160.35, 166.94, 173.85, 181.1, 187.74, 195.59, 202.73, 208.83, 218.58, 226.09, 232.23, 239.19, 245.54, 255.58, 266.33, 270.77, 277.34, 283.77, 291.46, 298.34, 305.59, 315.61, 323.27, 332.15, 338.91, 348.43, 352.62, 357.5, 362.96, 371.89, 383.89, 391.29, 397.79, 401.82, 410.14, 415.5, 426.7, 433.99, 441.52, 450.33, 456.52, 462.87, 475.64, 482.26, 493.08, 499.7, 509.78, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
    
    [20.17, 19.25, 20.94, 24.3, 25.84, 29.78, 28.21, 30.54, 31.61, 41.18, 44.74, 44.65, 40.9, 43.63, 40.09, 41.38, 54.23, 55.68, 57.91, 59.39, 61.56, 63.41, 64.26, 64.98, 66.2, 67.55, 69.28, 70.22, 71.77, 72.32, 74.11, 76.19, 97.25, 98.16, 98.99, 100.12, 101.05, 102.37, 102.91, 105.1, 106.31, 108.19, 108.78, 109.81, 118.74, 119.16, 120.84, 121.86, 130.21, 148.7, 130.03, 131.06, 132.96, 133.46, 135.61, 136.3, 137.62, 138.95, 139.26, 140.56, 142.34, 143.27, 144.83, 147.36, 167.26, 169.17, 169.57, 170.83, 172.04, 172.97, 174.4, 175.42, 176.16, 178.04, 179.11, 180.43, 181.97, 182.97, 182.8, 184.56, 198.77, 199.79, 199.83, 202.78, 204.25, 206.27, 208.55, 210.42, 212.73, 215.12, 216.57, 218.35, 219.79, 221.6, 223.94, 225.92, 242.88, 244.53, 246.3, 248.48],
    [34.88, 32.84, 33.67, 48.66, 51.85, 58.59, 52.21, 55.81, 57.01, 73.83, 78.67, 78.02, 69.23, 73.01, 68.44, 70.5, 99.5, 101.57, 104.2, 106.28, 109.11, 109.96, 111.42, 112.89, 114.43, 115.64, 118.12, 119.65, 120.99, 122.49, 123.72, 127.04, 168.82, 169.96, 171.4, 172.79, 174.48, 175.52, 176.92, 179.64, 180.52, 182.77, 184.57, 185.68, 196.61, 197.7, 199.2, 200.69, 222.88, 252.99, 225.09, 226.48, 228.33, 230.05, 232.07, 233.07, 234.72, 235.94, 237.52, 238.59, 241.09, 242.39, 244.51, 247.02, 287.99, 289.57, 290.75, 292.02, 293.31, 294.68, 296.96, 298.82, 299.84, 301.11, 302.88, 305.15, 306.7, 308.28, 310.12, 311.42, 341.73, 343.56, 343.68, 347.57, 351.24, 353.79, 357.22, 359.22, 362.76, 365.21, 367.23, 370.21, 372.27, 375.96, 378.39, 381.01, 415.89, 418.29, 420.54, 423.38],
    [36.55, 36.16, 36.49, 33.96, 31.85, 31.23, 32.19, 33.74, 37.23, 39.15, 41.05, 41.63, 38.78, 41.06, 46.93, 48.23, 54.83, 55.28, 57.43, 58.32, 61.24, 60.76, 61.66, 63.74, 66.52, 67.3, 72.08, 73.08, 75.15, 78.63, 81.06, 84.62, 93.33, 91.37, 93.11, 93.86, 95.91, 96.93, 98.06, 100.95, 103.09, 106.6, 107.76, 229.74, 232.48, 230.74, 231.71, 236.44, 238.82, 123.06, 120.95, 124.06, 125.81, 126.96, 133.24, 134.55, 136.86, 138.78, 141.26, 142.56, 144.92, 145.87, 149.86, 152.33, 159.39, 161.23, 158.38, 160.49, 161.49, 163.17, 165.04, 166.24, 168.9, 171.72, 173.05, 180.43, 181.66, 182.97, 178.85, 180.25, 185.93, 187.7, 203.35, 206.17, 208.11, 209.42, 210.9, 211.83, 215.34, 217.24, 215.82, 216.73, 218.71, 220.16, 222.86, 223.41, 238.65, 238.99, 241.29, 244.38],
    [67.63, 67.54, 70.04, 64.34, 64.57, 65.22, 59.98, 61.63, 66.32, 67.76, 74.41, 72.58, 72.78, 73.79, 76.49, 83.31, 91.27, 92.76, 96.22, 97.63, 102.38, 102.26, 103.11, 107.08, 109.5, 111.22, 118.8, 120.04, 123.41, 132.76, 136.28, 142.98, 161.88, 155.02, 156.24, 157.86, 162.38, 163.8, 165.31, 168.6, 172.59, 178.81, 180.82, 302.42, 305.4, 305.17, 306.3, 312.3, 315.39, 199.85, 197.84, 200.79, 204.38, 205.98, 213.13, 214.47, 218.11, 221.42, 223.41, 225.11, 228.7, 230.06, 238.82, 240.89, 257.69, 260.91, 277.29, 281.28, 264.47, 265.96, 270.36, 271.83, 274.46, 279.96, 281.46, 291.13, 292.73, 294.49, 290.11, 292.37, 300.36, 302.5, 318.97, 325.04, 326.39, 328.63, 331.85, 333.33, 339.25, 340.56, 342.45, 344.82, 346.76, 348.7, 355.16, 356.76, 381.08, 381.23, 384.29, 389.57],
    [55.82, 54.23, 55.02, 56.16, 57.02, 56.49, 51.39, 51.43, 51.06, 54.5, 55.4, 53.97, 55.43, 54.73, 56.55, 54.02, 53.45, 54.06, 54.57, 55.61, 57.92, 58.71, 60.53, 62.62, 65.3, 66.63, 69.93, 71.29, 73.05, 74.78, 76.41, 79.88, 84.0, 83.74, 84.38, 86.08, 87.48, 89.6, 90.8, 92.97, 94.49, 99.09, 100.66, 102.12, 103.61, 105.72, 106.82, 109.12, 113.35, 114.81, 116.35, 118.56, 120.55, 121.14, 126.58, 127.9, 130.23, 132.1, 133.5, 135.61, 139.33, 141.0, 144.5, 145.71, 149.05, 150.08, 152.56, 155.02, 156.86, 158.13, 159.85, 161.89, 163.9, 167.46, 169.06, 174.33, 175.88, 178.01, 174.42, 175.25, 184.4, 186.45, 202.77, 206.15, 208.28, 209.57, 211.1, 211.94, 216.02, 217.16, 228.95, 230.17, 232.88, 234.83, 238.23, 239.61, 248.33, 249.9, 250.59, 254.5],
    [93.04, 84.16, 80.98, 84.27, 83.05, 85.69, 83.52, 88.34, 92.28, 96.65, 104.7, 111.97, 110.47, 117.12, 123.7, 133.1, 146.05, 152.31, 160.35, 166.94, 173.85, 181.1, 187.74, 195.59, 202.73, 208.83, 218.58, 226.09, 232.23, 239.19, 245.54, 255.58, 266.33, 270.77, 277.34, 283.77, 291.46, 298.34, 305.59, 315.61, 323.27, 332.15, 338.91, 348.43, 352.62, 357.5, 362.96, 371.89, 383.89, 391.29, 397.79, 401.82, 410.14, 415.5, 426.7, 433.99, 441.52, 450.33, 456.52, 462.87, 475.64, 482.26, 493.08, 499.7, 509.78, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],

    [20.17, 19.25, 20.94, 24.3, 25.84, 29.78, 28.21, 30.54, 31.61, 41.18, 44.74, 44.65, 40.9, 43.63, 40.09, 41.38, 54.23, 55.68, 57.91, 59.39, 61.56, 63.41, 64.26, 64.98, 66.2, 67.55, 69.28, 70.22, 71.77, 72.32, 74.11, 76.19, 97.25, 98.16, 98.99, 100.12, 101.05, 102.37, 102.91, 105.1, 106.31, 108.19, 108.78, 109.81, 118.74, 119.16, 120.84, 121.86, 130.21, 148.7, 130.03, 131.06, 132.96, 133.46, 135.61, 136.3, 137.62, 138.95, 139.26, 140.56, 142.34, 143.27, 144.83, 147.36, 167.26, 169.17, 169.57, 170.83, 172.04, 172.97, 174.4, 175.42, 176.16, 178.04, 179.11, 180.43, 181.97, 182.97, 182.8, 184.56, 198.77, 199.79, 199.83, 202.78, 204.25, 206.27, 208.55, 210.42, 212.73, 215.12, 216.57, 218.35, 219.79, 221.6, 223.94, 225.92, 242.88, 244.53, 246.3, 248.48],
    [34.88, 32.84, 33.67, 48.66, 51.85, 58.59, 52.21, 55.81, 57.01, 73.83, 78.67, 78.02, 69.23, 73.01, 68.44, 70.5, 99.5, 101.57, 104.2, 106.28, 109.11, 109.96, 111.42, 112.89, 114.43, 115.64, 118.12, 119.65, 120.99, 122.49, 123.72, 127.04, 168.82, 169.96, 171.4, 172.79, 174.48, 175.52, 176.92, 179.64, 180.52, 182.77, 184.57, 185.68, 196.61, 197.7, 199.2, 200.69, 222.88, 252.99, 225.09, 226.48, 228.33, 230.05, 232.07, 233.07, 234.72, 235.94, 237.52, 238.59, 241.09, 242.39, 244.51, 247.02, 287.99, 289.57, 290.75, 292.02, 293.31, 294.68, 296.96, 298.82, 299.84, 301.11, 302.88, 305.15, 306.7, 308.28, 310.12, 311.42, 341.73, 343.56, 343.68, 347.57, 351.24, 353.79, 357.22, 359.22, 362.76, 365.21, 367.23, 370.21, 372.27, 375.96, 378.39, 381.01, 415.89, 418.29, 420.54, 423.38],
    [36.55, 36.16, 36.49, 33.96, 31.85, 31.23, 32.19, 33.74, 37.23, 39.15, 41.05, 41.63, 38.78, 41.06, 46.93, 48.23, 54.83, 55.28, 57.43, 58.32, 61.24, 60.76, 61.66, 63.74, 66.52, 67.3, 72.08, 73.08, 75.15, 78.63, 81.06, 84.62, 93.33, 91.37, 93.11, 93.86, 95.91, 96.93, 98.06, 100.95, 103.09, 106.6, 107.76, 229.74, 232.48, 230.74, 231.71, 236.44, 238.82, 123.06, 120.95, 124.06, 125.81, 126.96, 133.24, 134.55, 136.86, 138.78, 141.26, 142.56, 144.92, 145.87, 149.86, 152.33, 159.39, 161.23, 158.38, 160.49, 161.49, 163.17, 165.04, 166.24, 168.9, 171.72, 173.05, 180.43, 181.66, 182.97, 178.85, 180.25, 185.93, 187.7, 203.35, 206.17, 208.11, 209.42, 210.9, 211.83, 215.34, 217.24, 215.82, 216.73, 218.71, 220.16, 222.86, 223.41, 238.65, 238.99, 241.29, 244.38],
    [67.63, 67.54, 70.04, 64.34, 64.57, 65.22, 59.98, 61.63, 66.32, 67.76, 74.41, 72.58, 72.78, 73.79, 76.49, 83.31, 91.27, 92.76, 96.22, 97.63, 102.38, 102.26, 103.11, 107.08, 109.5, 111.22, 118.8, 120.04, 123.41, 132.76, 136.28, 142.98, 161.88, 155.02, 156.24, 157.86, 162.38, 163.8, 165.31, 168.6, 172.59, 178.81, 180.82, 302.42, 305.4, 305.17, 306.3, 312.3, 315.39, 199.85, 197.84, 200.79, 204.38, 205.98, 213.13, 214.47, 218.11, 221.42, 223.41, 225.11, 228.7, 230.06, 238.82, 240.89, 257.69, 260.91, 277.29, 281.28, 264.47, 265.96, 270.36, 271.83, 274.46, 279.96, 281.46, 291.13, 292.73, 294.49, 290.11, 292.37, 300.36, 302.5, 318.97, 325.04, 326.39, 328.63, 331.85, 333.33, 339.25, 340.56, 342.45, 344.82, 346.76, 348.7, 355.16, 356.76, 381.08, 381.23, 384.29, 389.57],
    [55.82, 54.23, 55.02, 56.16, 57.02, 56.49, 51.39, 51.43, 51.06, 54.5, 55.4, 53.97, 55.43, 54.73, 56.55, 54.02, 53.45, 54.06, 54.57, 55.61, 57.92, 58.71, 60.53, 62.62, 65.3, 66.63, 69.93, 71.29, 73.05, 74.78, 76.41, 79.88, 84.0, 83.74, 84.38, 86.08, 87.48, 89.6, 90.8, 92.97, 94.49, 99.09, 100.66, 102.12, 103.61, 105.72, 106.82, 109.12, 113.35, 114.81, 116.35, 118.56, 120.55, 121.14, 126.58, 127.9, 130.23, 132.1, 133.5, 135.61, 139.33, 141.0, 144.5, 145.71, 149.05, 150.08, 152.56, 155.02, 156.86, 158.13, 159.85, 161.89, 163.9, 167.46, 169.06, 174.33, 175.88, 178.01, 174.42, 175.25, 184.4, 186.45, 202.77, 206.15, 208.28, 209.57, 211.1, 211.94, 216.02, 217.16, 228.95, 230.17, 232.88, 234.83, 238.23, 239.61, 248.33, 249.9, 250.59, 254.5],
    [93.04, 84.16, 80.98, 84.27, 83.05, 85.69, 83.52, 88.34, 92.28, 96.65, 104.7, 111.97, 110.47, 117.12, 123.7, 133.1, 146.05, 152.31, 160.35, 166.94, 173.85, 181.1, 187.74, 195.59, 202.73, 208.83, 218.58, 226.09, 232.23, 239.19, 245.54, 255.58, 266.33, 270.77, 277.34, 283.77, 291.46, 298.34, 305.59, 315.61, 323.27, 332.15, 338.91, 348.43, 352.62, 357.5, 362.96, 371.89, 383.89, 391.29, 397.79, 401.82, 410.14, 415.5, 426.7, 433.99, 441.52, 450.33, 456.52, 462.87, 475.64, 482.26, 493.08, 499.7, 509.78, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],

]
inference_time_edge = [[y * 1.3 for y in x] for x in inference_time_edge]
inference_time_local = [
    [47.038, 215.772, 147.884, 508.42, 64.391, 244.08,     47.038, 215.772, 147.884, 508.42, 64.391, 244.08,     47.038, 215.772, 147.884, 508.42, 64.391, 244.08],
    [49.849, 215.44, 120.539, 499.914, 63.963, 427.169,    49.849, 215.44, 120.539, 499.914, 63.963, 427.169,    49.849, 215.44, 120.539, 499.914, 63.963, 427.169]
]
inference_time_edge = np.multiply(inference_time_edge, 0.001)
pred_fade = [2.35, 3.87, 3.95, 3.64, 3.85, 3.65, 3.65, 3.65, 3.65, 3.65, 3.65, 3.65, 4.89, 4.98, 4.77, 3.01, 2.8, 3.11, 1.4, 1.57, 1.87, 1.54, 1.63, 1.63, 3.24, 3.3, 2.93, 2.42, 2.28, 2.43, 2.27, 2.35, 2.35, 2.35, 2.35, 2.35, 2.35, 2.35, 2.35, 2.35, 1.58, 1.6, 1.77, 1.58, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 0.87, 0.92, 1.82, 1.62, 1.47, 1.68, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63, 1.63]
true_fade = [3.98, 3.98, 3.98, 3.98, 3.98, 3.98, 3.98, 3.98, 3.98, 3.98, 3.98, 5.55, 5.55, 5.55, 3.2, 3.2, 3.2, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 3.2, 3.2, 3.2, 2.42, 2.42, 2.42, 2.42, 2.42, 2.42, 2.42, 2.42, 2.42, 2.42, 2.42, 2.42, 2.42, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 0.86, 0.86, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64, 1.64]

pred_GPU = [0.23, 0.25, 0.27, 0.25, 0.25, 0.25, 0.25, 0.22, 0.22, 0.21, 0.21, 0.22, 0.25, 0.24, 0.25, 0.25, 0.26, 0.24, 0.23, 0.24, 0.25, 0.25, 0.24, 0.25, 0.25, 0.25, 0.25, 0.26, 0.26, 0.25, 0.26, 0.26, 0.26, 0.24, 0.25, 0.25, 0.26, 0.27, 0.27, 0.27, 0.28, 0.27, 0.26, 0.26, 0.24, 0.24, 0.23, 0.23, 0.23, 0.24, 0.22, 0.24, 0.22, 0.24, 0.24, 0.23, 0.23, 0.23, 0.24, 0.23, 0.23, 0.24, 0.26, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.28, 0.28, 0.28, 0.27, 0.27, 0.27, 0.27, 0.26, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.26, 0.27, 0.26, 0.26, 0.26, 0.26, 0.27, 0.27, 0.27, 0.27, 0.26, 0.27, 0.27, 0.27, 0.27, 0.27]
true_GPU = [0.25, 0.27, 0.24, 0.24, 0.25, 0.24, 0.2, 0.21, 0.2, 0.2, 0.21, 0.25, 0.24, 0.24, 0.25, 0.25, 0.23, 0.22, 0.24, 0.25, 0.24, 0.23, 0.24, 0.24, 0.25, 0.25, 0.25, 0.25, 0.24, 0.26, 0.25, 0.26, 0.23, 0.24, 0.24, 0.26, 0.27, 0.26, 0.26, 0.28, 0.26, 0.25, 0.26, 0.22, 0.23, 0.22, 0.22, 0.22, 0.23, 0.21, 0.23, 0.21, 0.23, 0.23, 0.22, 0.22, 0.22, 0.24, 0.22, 0.22, 0.23, 0.26, 0.27, 0.27, 0.27, 0.27, 0.26, 0.27, 0.28, 0.27, 
0.27, 0.27, 0.26, 0.26, 0.27, 0.25, 0.27, 0.26, 0.26, 0.27, 0.27, 0.27, 0.27, 0.25, 0.26, 0.26, 0.25, 0.26, 0.26, 0.27, 0.27, 0.26, 0.27, 0.25, 0.27, 0.26, 0.26, 0.26, 0.27, 0.27]
