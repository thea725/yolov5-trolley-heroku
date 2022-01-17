from streamlit_webrtc import ClientSettings

CLASSES = [ 'pepsodent', 'UC lemon', 'UC orange' ,'yakult' ]


WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )