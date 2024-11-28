from datetime import datetime

try:
    import cv2
    import pandas as pd
    import utils
    import hashlib
except ImportError:
    import auto_start

    auto_start.setup_environment()
finally:
    import cv2
    import pandas as pd
    import utils
    import hashlib

__cap = cv2.VideoCapture(1)

__processor_name = utils.get_processor_info()


def _get_filename(len_df: int):
    return (f"./csv/{hashlib.md5(__processor_name.encode()).hexdigest()}"
            f"_EAR_{len_df}_{datetime.now().strftime('%B_%d_%H_%M_%S')}.csv")


def main():
    dataframe = pd.DataFrame(columns=['lefr_aer', 'right_ear', 'timestamp'])
    while True:
        try:
            _, frame = __cap.read()
            ear_value = utils.get_ear_from_image(frame)
            if ear_value:
                left, right = ear_value
                dataframe.loc[-1] = [left, right, datetime.now()]
                dataframe.index = dataframe.index + 1
                dataframe = dataframe.sort_index()

                if len(dataframe) > 10_000:
                    df_name = _get_filename(len(dataframe))
                    dataframe.to_csv(df_name, index=False)
                    dataframe = dataframe.iloc[0:0]

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:
            __cap.release()
            cv2.destroyAllWindows()
            df_name = _get_filename(len(dataframe))
            dataframe.to_csv(df_name, index=False)
            break


if __name__ == '__main__':
    main()
