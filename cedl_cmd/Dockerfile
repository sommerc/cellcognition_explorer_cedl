FROM floydhub/dl-docker:cpu 

COPY cedl.py /root/cedl.py
COPY autoencoders.py /root/
COPY utils.py /root/

RUN pip install https://github.com/Lasagne/Lasagne/archive/master.zip
RUN pip install h5py --upgrade
RUN pip install git+https://github.com/Cellh5/cellh5@master -t /root/
RUN pip install nolearn
RUN echo install | pip install git+https://github.com/lisa-lab/pylearn2@master -t /root/ --no-deps


ENTRYPOINT  ["python", "/root/cedl.py"]





